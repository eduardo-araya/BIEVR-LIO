#ifndef BIEVR_LIO_CONFIG_LOADER_H_
#define BIEVR_LIO_CONFIG_LOADER_H_

// Single source of truth for the wrapper configuration. This header is
// ROS-agnostic: it reads plain YAML files via yaml-cpp and fills `Config`, so
// the ROS1 and ROS2 wrappers share one loader and one set of YAML config files.
// The ROS layer only carries the *paths* of the files to load.
//
// The YAML is organised into sections (topics / calibration / lidar / imu /
// map / preprocess / optimization / debug). Files are
// layered: callers pass {params, sensor_config} and the later file wins on a
// per-leaf basis, so the sensor config wins on any leaf both files define and a
// section that appears in both (e.g. `imu`) merges by key.
//
// Consumers need yaml-cpp on their include/link path (header-only here; the core
// library itself does not compile or depend on it).

#include <bievr_lio/bievr_map.h>
#include <bievr_lio/common.h>
#include <bievr_lio/log++.h>
#include <bievr_lio/pipeline.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace bievr {

struct TopicConfig {
  std::string pointcloud_topic = "/points";
  std::string imu_topic = "/imu";
  std::string bag_path = "";
};

// Input-side Livox noise rejection, applied during message -> internal-cloud
// conversion (before range filtering / undistortion). The Livox per-point `tag`
// byte encodes: bits 0-1 spatial-position noise confidence (0 normal, 1
// high-confidence noise, 2 medium, 3 low — water/mirror reflections get flagged
// here), bits 2-3 intensity-based noise confidence, bits 4-5 echo number.
// Structural no-op on clouds without tag/intensity fields (e.g. simulators).
struct TagFilterConfig {
  bool enable = false;
  std::vector<int> spatial_allowed = {0};
  std::vector<int> intensity_allowed = {0};
  std::vector<int> echo_allowed = {0, 1};
};

struct InputFilterConfig {
  TagFilterConfig tag_filter;
  double min_intensity = 0.0;  // drop points with intensity < this; <= 0 disables
};

// Precomputes the 256-entry keep/drop table for the Livox tag byte from the
// three allow-lists, so the per-point test is a single array lookup.
inline std::array<uint8_t, 256> makeTagKeepLut(const TagFilterConfig& config) {
  auto allowed = [](const std::vector<int>& list, int v) {
    return std::find(list.begin(), list.end(), v) != list.end();
  };
  std::array<uint8_t, 256> keep{};
  for (int tag = 0; tag < 256; ++tag) {
    const int spatial = tag & 0x3;
    const int intensity = (tag >> 2) & 0x3;
    const int echo = (tag >> 4) & 0x3;
    keep[tag] = allowed(config.spatial_allowed, spatial) &&
                allowed(config.intensity_allowed, intensity) && allowed(config.echo_allowed, echo);
  }
  return keep;
}

// One virtual filter box, anchored to a TF frame (e.g. a URDF link). Points of
// every incoming scan that fall inside any box are removed before the pipeline
// sees them (self-filtering of the carrier vehicle's own body).
struct SelfFilterBoxConfig {
  std::string name;                 // label (stats / markers)
  std::string frame_id;             // TF frame the box is anchored to
  V3 size = V3::Zero();             // full extents [m] in the box frame
  V3 offset_translation = V3::Zero();
  V3 offset_rpy = V3::Zero();       // box pose in frame_id; R = Rz(y)*Ry(p)*Rx(r)
};

struct SelfFilterConfig {
  bool enable = false;
  bool visualize = false;           // publish RViz markers of the boxes
  double visualize_rate_hz = 5.0;
  std::vector<SelfFilterBoxConfig> boxes;
};

// Output re-expression: when `base_frame` is non-empty, the wrapper publishes
// odometry/TF for `base_frame` instead of the native IMU body frame, resolving
// TF(sensor_frame <- base_frame) live per publish. This supports carriers where
// the sensor is NOT rigidly attached to the base (e.g. sensor on a rotating
// upper body, base on the chassis, joined by a continuous revolute joint).
struct FramesConfig {
  std::string base_frame = "";      // "" = native behavior (odometry in IMU frame)
  std::string sensor_frame = "";    // TF frame the raw points arrive in (LiDAR link)
  double tf_buffer_seconds = 2.0;
  int max_tf_age_ms = 100;          // cached-TF fallback validity window
};

struct OutputConfig {
  // When > 0, odometry/TF are ALSO published from the IMU-propagation path,
  // throttled to this rate using EVENT timestamps (sim/bag safe). 0 = publish
  // only per processed LiDAR frame (native behavior).
  double odom_rate_hz = 0.0;
  // Accumulated map cloud for RViz/debug (BIEVR's map is a bump-image voxel
  // structure with no native cloud output). 0 = off.
  double map_cloud_rate_hz = 1.0;
  double map_cloud_voxel_m = 0.2;
};

struct Config {
  TopicConfig topic_config;
  Pipeline::Config pipeline_config;
  InputFilterConfig input_filter;
  SelfFilterConfig self_filter;
  FramesConfig frames;
  OutputConfig output;
  // Upper bound on threads used for TBB parallel regions. 0 = let TBB decide
  // (use all available cores); >0 caps parallelism to that many threads.
  int max_num_threads = 0;
};

namespace config_internal {

// A read-only view over one or more YAML documents. Lookups address a
// `section.key` pair and scan the documents in reverse, so the last file that
// defines that exact leaf wins (sections shared across files merge per key).
class MergedYaml {
 public:
  void add(const YAML::Node& node) { nodes_.push_back(node); }

  template <typename T>
  T get(const std::string& section, const std::string& key, const T& default_value) const {
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      const YAML::Node& root = *it;
      if (root[section] && root[section][key]) {
        return root[section][key].as<T>();
      }
    }
    return default_value;
  }

  // Same as get(), but for a key that lives at the document root (no section).
  template <typename T>
  T getTopLevel(const std::string& key, const T& default_value) const {
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      const YAML::Node& root = *it;
      if (root[key]) {
        return root[key].as<T>();
      }
    }
    return default_value;
  }

  // Raw access to a `section.key` node (e.g. a list) — the whole node comes from
  // the last file that defines it (no per-element merging). Invalid node if none.
  YAML::Node getRaw(const std::string& section, const std::string& key) const {
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      const YAML::Node& root = *it;
      if (root[section] && root[section][key]) {
        return root[section][key];
      }
    }
    return YAML::Node(YAML::NodeType::Undefined);
  }

  // Nested lookup: section.sub.key (used for blocks like frames.tf.*).
  template <typename T>
  T getNested(const std::string& section, const std::string& sub, const std::string& key,
              const T& default_value) const {
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      const YAML::Node& root = *it;
      if (root[section] && root[section][sub] && root[section][sub][key]) {
        return root[section][sub][key].as<T>();
      }
    }
    return default_value;
  }

 private:
  std::vector<YAML::Node> nodes_;
};

// Builds a Transform from a section holding `translation` (3) and `rotation` (9)
// vectors. Extrinsics are mandatory: a missing section yields empty vectors and
// an incomplete section yields wrong-sized ones, both of which are hard errors
// (no silent fallback to identity). Returns false and leaves `out` untouched on
// failure.
inline bool extrinsicFromVectors(const std::vector<double>& t_vec, const std::vector<double>& R_vec,
                                 const std::string& label, Transform& out) {
  if (t_vec.size() != 3) {
    LOG(E, "Config error: " << label << " translation must have 3 elements, got " << t_vec.size()
                            << ". Extrinsics must be provided completely.");
    return false;
  }
  if (R_vec.size() != 9) {
    LOG(E, "Config error: " << label << " rotation must have 9 elements, got " << R_vec.size()
                            << ". Extrinsics must be provided completely.");
    return false;
  }
  const V3 t(t_vec[0], t_vec[1], t_vec[2]);
  Rotation R;
  R << R_vec[0], R_vec[1], R_vec[2], R_vec[3], R_vec[4], R_vec[5], R_vec[6], R_vec[7], R_vec[8];
  out = Transform(R, t);
  return true;
}

// Reads a scalar that is allowed to be absent (the positive `default_value` is
// used then) but, when present, must be positive. A non-positive value is a hard
// error: reports it and returns false. On success writes the value to `out`.
template <typename T>
bool getPositive(const MergedYaml& yaml, const std::string& section, const std::string& key,
                 const T& default_value, T& out) {
  out = yaml.get<T>(section, key, default_value);
  if (out <= T(0)) {
    LOG(E, "Config error: '" << section << "." << key << "' must be positive, got " << out << ".");
    return false;
  }
  return true;
}

// Appends a transform as a translation vector and a flat row-major rotation, so
// the overview stays on a couple of lines per extrinsic.
inline void printExtrinsic(std::ostream& os, const std::string& label, const Transform& T) {
  const auto& t = T.translation();
  const auto& R = T.rotation();
  os << "  " << label << " t:  [" << t.x() << ", " << t.y() << ", " << t.z() << "]\n";
  os << "  " << label << " R:  [" << R(0, 0) << ", " << R(0, 1) << ", " << R(0, 2) << ", "
     << R(1, 0) << ", " << R(1, 1) << ", " << R(1, 2) << ", " << R(2, 0) << ", " << R(2, 1) << ", "
     << R(2, 2) << "]\n";
}

// Dumps every parameter the loader resolved (defaults included), so the run log
// shows exactly what configuration was used.
inline void printConfigOverview(const Config& config) {
  const auto& tc = config.topic_config;
  const auto& hc = config.pipeline_config;
  auto yn = [](bool b) { return b ? "true" : "false"; };

  std::ostringstream os;
  os << "================ BIEVR-LIO config ================\n";
  os << "topics:\n";
  os << "  pointcloud:           " << tc.pointcloud_topic << "\n";
  os << "  imu:                  " << tc.imu_topic << "\n";
  os << "max_num_threads:        "
     << (config.max_num_threads > 0 ? std::to_string(config.max_num_threads)
                                    : std::string("automatic"))
     << "\n";
  os << "lidar:\n";
  os << "  min_range_m:          " << hc.preprocess.min_range << "\n";
  os << "  max_range_m:          " << hc.preprocess.max_range << "\n";
  os << "map:\n";
  os << "  pixel_size_m:   " << hc.map.px_size << "\n";
  os << "  voxel_size_m:   " << hc.map.voxel_size << "\n";
  os << "  normal_tolerance_deg: " << hc.map.norm_tol_deg << "\n";
  os << "  max_size:             " << hc.map.max_size << "\n";
  os << "  smooth:               " << yn(hc.map.smooth) << "\n";
  os << "  weighted:             " << yn(hc.map.weighted) << "\n";
  os << "  frame:                " << hc.map_frame << "\n";
  os << "preprocess:\n";
  os << "  downsample_res_m:     " << hc.preprocess.downsample_resolution << "\n";
  os << "  informed_sampling:    " << yn(hc.preprocess.informed_sampling) << "\n";
  os << "optimization:\n";
  os << "  huber_delta:          " << hc.registration.huber_delta << "\n";
  os << "  img_residual:         " << yn(hc.registration.img_residual) << "\n";
  os << "  img_jacobian:         " << yn(hc.registration.img_jacobian) << "\n";
  os << "imu:\n";
  os << "  window_s:             " << hc.imu.window_length_s << "\n";
  os << "  t_init:               " << hc.imu.t_init << "\n";
  os << "  normalized:           " << hc.imu.normalized << "\n";
  os << "  frame:                " << hc.body_frame << "\n";
  os << "debug:\n";
  os << "  publish_all_clouds:        " << yn(hc.publish_all_clouds) << "\n";
  os << "  print_timing:         " << yn(hc.print_timing) << "\n";
  os << "  print_debug:          " << yn(hc.print_debug) << "\n";
  os << "  dashboard:            " << yn(hc.print_dashboard) << "\n";
  if (hc.print_dashboard) {
    os << "  dashboard_ascii_path: "
       << (hc.dashboard_ascii_path.empty() ? "<none>" : hc.dashboard_ascii_path) << "\n";
  }
  os << "  log_path:             " << (hc.log_path.empty() ? "<none>" : hc.log_path) << "\n";
  os << "calibration (LiDAR -> IMU):\n";
  printExtrinsic(os, "T_I_L", hc.T_I_L);
  os << "input_filter:\n";
  os << "  tag_filter:           " << yn(config.input_filter.tag_filter.enable) << "\n";
  os << "  min_intensity:        " << config.input_filter.min_intensity << "\n";
  os << "self_filter:            " << yn(config.self_filter.enable) << " ("
     << config.self_filter.boxes.size() << " boxes)\n";
  os << "frames:\n";
  os << "  base_frame:           "
     << (config.frames.base_frame.empty() ? "<native body frame>" : config.frames.base_frame)
     << "\n";
  os << "  sensor_frame:         "
     << (config.frames.sensor_frame.empty() ? "<none>" : config.frames.sensor_frame) << "\n";
  os << "output:\n";
  os << "  odom_rate_hz:         " << config.output.odom_rate_hz << "\n";
  os << "==================================================";
  LOG(I, os.str());
}

}  // namespace config_internal

// Loads `Config` from one or more plain YAML files (no ROS parameter server).
// Later files override earlier ones per leaf key. Returns false if a file cannot
// be opened or parsed.
inline bool loadConfigFromYaml(const std::vector<std::string>& yaml_paths, Config& config) {
  config_internal::MergedYaml yaml;
  for (const std::string& path : yaml_paths) {
    if (path.empty()) {
      continue;
    }
    LOG(I, "Loading config from '" << path << "'.");
    try {
      yaml.add(YAML::LoadFile(path));
    } catch (const std::exception& e) {
      LOG(E, "Failed to load YAML config '" << path << "': " << e.what());
      return false;
    }
  }

  auto& tc = config.topic_config;
  auto& hc = config.pipeline_config;

  // --- topics ---
  tc.pointcloud_topic = yaml.get<std::string>("topics", "pointcloud", tc.pointcloud_topic);
  tc.imu_topic = yaml.get<std::string>("topics", "imu", tc.imu_topic);

  // --- lidar ---
  hc.preprocess.min_range = yaml.get<double>("lidar", "min_range_m", 0.5);
  hc.preprocess.max_range = yaml.get<double>("lidar", "max_range_m", 100.);

  // --- map ---
  // These resolutions/tolerances are optional (fall back to the defaults below)
  // but must be positive when supplied; a non-positive value aborts the load.
  int max_size = 0;
  if (!config_internal::getPositive(yaml, "map", "pixel_size_m", 0.05, hc.map.px_size) ||
      !config_internal::getPositive(yaml, "map", "voxel_size_m", 0.5, hc.map.voxel_size) ||
      !config_internal::getPositive(yaml, "map", "normal_tolerance_deg", 3., hc.map.norm_tol_deg) ||
      !config_internal::getPositive(yaml, "map", "max_size", 5000000, max_size)) {
    return false;
  }
  hc.map.max_size = static_cast<size_t>(max_size);
  hc.map.smooth = yaml.get<bool>("map", "smooth", false);
  hc.map.weighted = yaml.get<bool>("map", "weighted", false);
  // The map frame is the parent (odometry) frame for published poses/clouds.
  hc.map_frame = yaml.get<std::string>("map", "frame", hc.map_frame);

  // --- preprocess ---
  if (!config_internal::getPositive(yaml, "preprocess", "downsample_resolution_m", 0.15,
                                    hc.preprocess.downsample_resolution)) {
    return false;
  }
  hc.preprocess.informed_sampling = yaml.get<bool>("preprocess", "informed_sampling", false);

  // --- optimization ---
  if (!config_internal::getPositive(yaml, "optimization", "huber_delta", 100.,
                                    hc.registration.huber_delta)) {
    return false;
  }
  hc.registration.img_residual = yaml.get<bool>("optimization", "img_residual", true);
  hc.registration.img_jacobian = yaml.get<bool>("optimization", "img_jacobian", true);

  // --- imu (params side: inertial window + normalization) ---
  if (!config_internal::getPositive(yaml, "imu", "window_s", 10., hc.imu.window_length_s) ||
      !config_internal::getPositive(yaml, "imu", "t_init", 0.2, hc.imu.t_init)) {
    return false;
  }
  // Accelerometer normalization, resolved during bias estimation:
  //   < 0 autodetect, = 0 not normalized, > 0 normalized (multiplied by g)
  hc.imu.normalized = yaml.get<double>("imu", "normalized", -1.0);
  // The IMU frame is the body (child) frame of the published odometry.
  hc.body_frame = yaml.get<std::string>("imu", "frame", hc.body_frame);

  // --- debug ---
  hc.publish_all_clouds = yaml.get<bool>("debug", "publish_all_clouds", false);
  hc.print_timing = yaml.get<bool>("debug", "timing", false);
  hc.print_debug = yaml.get<bool>("debug", "log", false);
  hc.log_path = yaml.get<std::string>("debug", "trajectory_path", "");

  // --- dashboard (live status print) ---
  hc.print_dashboard = yaml.get<bool>("debug", "dashboard", false);
  hc.dashboard_ascii_path = yaml.get<std::string>("debug", "dashboard_ascii_path", "");
  // If the dashboard is enabled but no explicit art path is given, look for
  // `bievr_ascii.txt` next to the config files that were loaded. Those files may
  // live in different directories (e.g. params.yaml in config/, the sensor
  // config in config/sensor_configs/), so probe each config's directory and its
  // parent and take the first that actually contains the file rather than
  // guessing a path that may not exist.
  if (hc.print_dashboard && hc.dashboard_ascii_path.empty()) {
    namespace fs = std::filesystem;
    for (const std::string& p : yaml_paths) {
      if (p.empty()) continue;
      const fs::path dir = fs::path(p).parent_path();
      for (const fs::path& candidate_dir : {dir, dir.parent_path()}) {
        const fs::path candidate = candidate_dir / "bievr_ascii.txt";
        std::error_code ec;
        if (fs::exists(candidate, ec)) {
          hc.dashboard_ascii_path = candidate.string();
          break;
        }
      }
      if (!hc.dashboard_ascii_path.empty()) break;
    }
  }

  // --- lidar_input_filter (Livox tag + intensity noise rejection) ---
  auto& fc = config.input_filter;
  fc.tag_filter.enable = yaml.getNested<bool>("lidar_input_filter", "tag_filter", "enable", false);
  fc.tag_filter.spatial_allowed = yaml.getNested<std::vector<int>>(
      "lidar_input_filter", "tag_filter", "spatial_allowed", fc.tag_filter.spatial_allowed);
  fc.tag_filter.intensity_allowed = yaml.getNested<std::vector<int>>(
      "lidar_input_filter", "tag_filter", "intensity_allowed", fc.tag_filter.intensity_allowed);
  fc.tag_filter.echo_allowed = yaml.getNested<std::vector<int>>(
      "lidar_input_filter", "tag_filter", "echo_allowed", fc.tag_filter.echo_allowed);
  fc.min_intensity = yaml.get<double>("lidar_input_filter", "min_intensity", 0.0);

  // --- self_filter (virtual boxes anchored to TF frames) ---
  auto& sf = config.self_filter;
  sf.enable = yaml.get<bool>("self_filter", "enable", false);
  sf.visualize = yaml.get<bool>("self_filter", "visualize", false);
  sf.visualize_rate_hz = yaml.get<double>("self_filter", "visualize_rate_hz", 5.0);
  if (const YAML::Node boxes = yaml.getRaw("self_filter", "boxes"); boxes.IsSequence()) {
    for (const YAML::Node& b : boxes) {
      SelfFilterBoxConfig box;
      box.name = b["name"] ? b["name"].as<std::string>() : "";
      box.frame_id = b["frame_id"] ? b["frame_id"].as<std::string>() : "";
      const auto size = b["size"] ? b["size"].as<std::vector<double>>() : std::vector<double>{};
      if (box.frame_id.empty() || size.size() != 3) {
        LOG(E, "Config error: self_filter box '" << box.name
                                                 << "' needs a frame_id and a 3-element size.");
        return false;
      }
      box.size = V3(size[0], size[1], size[2]);
      if (const YAML::Node off = b["offset"]) {
        if (const YAML::Node t = off["translation"]) {
          const auto v = t.as<std::vector<double>>();
          if (v.size() == 3) box.offset_translation = V3(v[0], v[1], v[2]);
        }
        if (const YAML::Node r = off["rotation_rpy"]) {
          const auto v = r.as<std::vector<double>>();
          if (v.size() == 3) box.offset_rpy = V3(v[0], v[1], v[2]);
        }
      }
      sf.boxes.push_back(box);
    }
  }
  if (sf.enable && sf.boxes.empty()) {
    LOG(W, "self_filter.enable is true but no boxes are configured; disabling.");
    sf.enable = false;
  }

  // --- frames (non-rigid base-frame output re-expression) ---
  auto& fr = config.frames;
  fr.base_frame = yaml.get<std::string>("frames", "base_frame", "");
  fr.sensor_frame = yaml.get<std::string>("frames", "sensor_frame", "");
  fr.tf_buffer_seconds = yaml.getNested<double>("frames", "tf", "buffer_seconds", 2.0);
  fr.max_tf_age_ms = yaml.getNested<int>("frames", "tf", "max_tf_age_ms", 100);
  if (!fr.base_frame.empty() && fr.sensor_frame.empty()) {
    LOG(E, "Config error: frames.base_frame requires frames.sensor_frame (the TF frame the raw "
           "points arrive in).");
    return false;
  }

  // --- output (high-rate IMU-propagated odometry + debug map cloud) ---
  config.output.odom_rate_hz = yaml.get<double>("output", "odom_rate_hz", 0.0);
  config.output.map_cloud_rate_hz = yaml.get<double>("output", "map_cloud_rate_hz", 1.0);
  config.output.map_cloud_voxel_m = yaml.get<double>("output", "map_cloud_voxel_m", 0.2);

  // Lower the log level so DEBUG messages are shown when requested, otherwise
  // keep the default (INFO and above).
  LOG_SET_LEVEL(hc.print_debug ? BaseSeverity::DEBUG : BaseSeverity::INFO);

  // --- calibration (LiDAR -> IMU) ---
  if (!config_internal::extrinsicFromVectors(
          yaml.get<std::vector<double>>("calibration", "translation", {}),
          yaml.get<std::vector<double>>("calibration", "rotation", {}), "calibration", hc.T_I_L)) {
    return false;
  }

  // --- threading (top-level, process-wide) ---
  config.max_num_threads = yaml.getTopLevel<int>("max_num_threads", 0);
  if (config.max_num_threads < 0) {
    LOG(E, "Config error: 'max_num_threads' must be >= 0 (0 = automatic), got "
               << config.max_num_threads << ".");
    return false;
  }

  config_internal::printConfigOverview(config);
  return true;
}

// Loads `Config` from the executable's command-line arguments (the launch file
// passes the YAML paths as argv rather than ROS parameters, so this is fully
// ROS-agnostic and shared by both wrappers). Recognised flags:
//   --sensor_config_file <path>   --params_file <path>   --bag <path>
// `args` is the argument list with any ROS-specific arguments already stripped
// (ros::init does this in place on ROS1; rclcpp::remove_ros_arguments on ROS2).
inline bool loadConfigFromArgs(const std::vector<std::string>& args, Config& config) {
  std::string sensor_config_file;
  std::string params_file;
  std::string bag_path;
  bool have_bag = false;

  for (size_t i = 1; i < args.size(); ++i) {
    auto value = [&](const char* flag) -> std::string {
      if (i + 1 >= args.size()) {
        LOG(E, "Missing value after " << flag << ".");
        return "";
      }
      return args[++i];
    };
    if (args[i] == "--sensor_config_file") {
      sensor_config_file = value("--sensor_config_file");
    } else if (args[i] == "--params_file") {
      params_file = value("--params_file");
    } else if (args[i] == "--bag") {
      bag_path = value("--bag");
      have_bag = true;
    } else {
      LOG(W, "Ignoring unrecognized argument '" << args[i] << "'.");
    }
  }

  // Later files override earlier ones: params first, then sensor config, so the
  // sensor file wins on any leaf the two files share.
  if (!loadConfigFromYaml({params_file, sensor_config_file}, config)) {
    return false;
  }
  // The bag path is a launch argument, not part of the shared YAML config; only
  // override the config value when it was actually passed on the command line.
  if (have_bag) {
    config.topic_config.bag_path = bag_path;
  }
  return true;
}

}  // namespace bievr

#endif  // BIEVR_LIO_CONFIG_LOADER_H_
