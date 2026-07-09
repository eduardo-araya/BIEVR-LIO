#ifndef BIEVR_LIO_ROS2_OUTPUT_MANAGER_H_
#define BIEVR_LIO_ROS2_OUTPUT_MANAGER_H_

// Odometry output stage for the ROS2 wrapper. Replaces the pipeline's native
// Odometry publisher registration to provide two features the core does not
// have:
//
//  1. Non-rigid base-frame output. The pipeline estimates T_world_imu; when
//     `frames.base_frame` is configured, every published state is re-expressed
//     as T_world_base = T_world_imu * T_imu_lidar * TF(lidar <- base) with the
//     TF resolved live at the output stamp. This supports carriers where the
//     sensor is NOT rigidly attached to the base (sensor on a rotating upper
//     body, base_link on the chassis, continuous revolute joint in between).
//     TF lookup ladder: exact stamp -> latest available -> cached
//     (<= max_tf_age_ms). If nothing is usable the publish is skipped —
//     never fall back to publishing the sensor pose as the base pose (the
//     sensor link would get two TF parents).
//
//  2. High-rate output (`output.odom_rate_hz`). Between LiDAR corrections the
//     latest corrected state is propagated on every incoming IMU sample
//     (gyro attitude + constant world-frame velocity) and published through
//     the same conversion path, throttled by a scheduled-anchor scheme on
//     EVENT time (sim/bag safe): the anchor advances by exactly one period
//     per publish, clamped to at most one period behind, and re-anchors on
//     backward time jumps. Scan-synchronized publishes always happen and
//     reset the throttle window.
//
//  3. Odom anchored at the BASE start pose. BIEVR's
//     native world frame sits at the initial IMU pose — the sensor can sit
//     well above the base with an arbitrary yaw, which renders the
//     2D grid at sensor height and puts base_link nowhere near the odom
//     origin. On the first successful base conversion a constant anchor
//     T_world_anchor (yaw + translation of the initial base pose; roll/pitch
//     excluded so odom stays gravity-aligned) is latched, and EVERY output —
//     odometry, TF, and all world-frame clouds — is re-expressed through
//     T_anchor_inv. Because the clouds are transformed in the same process,
//     TF and cloud contents can never disagree. The anchor is constant for
//     the whole run (this is a choice of world origin at init, NOT continuous
//     re-anchoring).
//
//  4. Map cloud for RViz/debug (`output.map_cloud_rate_hz`): BIEVR's map is a
//     bump-image voxel structure with no native cloud, so the wrapper
//     accumulates the (anchored) registered scans into a voxel-downsampled
//     cloud and publishes it on `map` at a low rate.
//
// Threading: everything here runs on the single rclcpp spin thread (the
// synchronizer calls Pipeline::processFrame inline from the subscriber
// callbacks), so no locking is needed.

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>

#include <cmath>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include "bievr_lio/common.h"
#include "bievr_lio/config_loader.h"
#include "bievr_lio/log++.h"
#include "bievr_lio/pipeline.h"
#include "bievr_lio/preprocess.h"
#include "bievr_lio_ros2/publisher.h"

namespace bievr {

class OutputManager {
 public:
  OutputManager(rclcpp::Node::SharedPtr node, std::shared_ptr<tf2_ros::Buffer> tf_buffer,
                const Config& config)
      : node_(node),
        tf_buffer_(std::move(tf_buffer)),
        frames_(config.frames),
        world_frame_(config.pipeline_config.map_frame),
        body_frame_(config.pipeline_config.body_frame),
        T_I_L_(config.pipeline_config.T_I_L),
        period_ns_(config.output.odom_rate_hz > 0.0
                       ? static_cast<uint64_t>(1e9 / config.output.odom_rate_hz)
                       : 0),
        map_cloud_period_ns_(config.output.map_cloud_rate_hz > 0.0
                                 ? static_cast<uint64_t>(1e9 / config.output.map_cloud_rate_hz)
                                 : 0),
        map_cloud_voxel_(config.output.map_cloud_voxel_m),
        odom_pub_(node->create_publisher<nav_msgs::msg::Odometry>("odom", rclcpp::QoS(10))),
        tf_broadcaster_(std::make_shared<tf2_ros::TransformBroadcaster>(node)) {}

  // Takes over the pipeline's Odometry and cloud publishing; all other types
  // keep going through the native publisher.
  void registerWith(std::shared_ptr<Pipeline> pipeline, std::shared_ptr<Publisher> native) {
    native_ = native;
    pipeline->registerPublisher<Odometry>(
        [this](const Odometry& odom, const Header& header, const std::string& topic,
               const std::string& child_frame) { onOdometry(odom, header, topic, child_frame); });
    // Intercept the bias vectors to de-bias the gyro propagation; forward them
    // so the native bias topics stay alive.
    pipeline->registerPublisher<V3>([this](const V3& vec, const Header& header,
                                           const std::string& topic,
                                           const std::string& child_frame) {
      if (topic == "bias/gyro") {
        gyro_bias_ = vec;
      }
      if (auto native = native_.lock()) {
        native->publish(vec, header, topic, child_frame);
      }
    });
    // World-frame clouds are re-expressed through the odom anchor so they stay
    // consistent with the published TF; body-frame debug clouds pass through.
    pipeline->registerPublisher<IntensityPointcloud>(
        [this](const IntensityPointcloud& cloud, const Header& header, const std::string& topic,
               const std::string& child_frame) {
          onCloud<IntensityPointcloud>(cloud, header, topic, child_frame);
        });
    pipeline->registerPublisher<Pointcloud>(
        [this](const Pointcloud& cloud, const Header& header, const std::string& topic,
               const std::string& child_frame) {
          onCloud<Pointcloud>(cloud, header, topic, child_frame);
        });
  }

  // Call from the IMU subscription AFTER the synchronizer (so a scan completed
  // by this sample corrects the state first).
  void onImu(const ImuMeasurement& imu) {
    if (period_ns_ == 0 || !have_state_) {
      return;
    }
    // Backward jump (bag loop / restart): re-anchor and wait for a fresh state.
    if (imu.stamp + 1'000'000'000ULL < prop_stamp_) {
      LOG(W, "[Output] backward time jump detected; re-anchoring the odometry throttle.");
      next_pub_ns_ = imu.stamp;
      have_state_ = false;
      return;
    }
    if (imu.stamp <= prop_stamp_) {
      return;
    }

    // Propagate: gyro attitude + constant world-frame velocity. Accelerometer
    // integration is intentionally omitted — biases/gravity live inside the
    // core optimizer and the state re-corrects at every LiDAR frame (~10 Hz),
    // so the constant-velocity error over <=100 ms is negligible for control.
    const double dt = static_cast<double>(imu.stamp - prop_stamp_) * 1e-9;
    const V3 w = (imu.gyro - gyro_bias_) * dt;
    const double angle = w.norm();
    if (angle > 1e-12) {
      T_W_I_.linear() = T_W_I_.rotation() * Eigen::AngleAxisd(angle, w / angle).toRotationMatrix();
    }
    T_W_I_.translation() += v_W_ * dt;
    prop_stamp_ = imu.stamp;
    gyro_I_ = imu.gyro - gyro_bias_;

    // Scheduled-anchor throttle on event time.
    if (imu.stamp >= next_pub_ns_) {
      publishState(imu.stamp);
      next_pub_ns_ += period_ns_;
      if (next_pub_ns_ + period_ns_ < imu.stamp) {
        next_pub_ns_ = imu.stamp - period_ns_;  // at most one period behind
      }
    }
  }

 private:
  // Scan-corrected state from the pipeline (published once per LiDAR frame).
  void onOdometry(const Odometry& odom, const Header& header, const std::string& /*topic*/,
                  const std::string& /*child_frame*/) {
    T_W_I_ = odom.pose;
    v_W_ = odom.pose.rotation() * odom.linear_velocity;  // twist arrives body-frame
    gyro_I_ = odom.angular_velocity - gyro_bias_;
    prop_stamp_ = header.stamp;
    have_state_ = true;

    publishState(header.stamp);
    next_pub_ns_ = header.stamp + period_ns_;  // scan publish resets the window

    // Keep the native sensor-frame odometry as a debug topic (message only; the
    // native handler would also broadcast world->imu TF, which must NOT happen —
    // the sensor link already has a TF parent via the robot description).
    if (auto native = native_.lock()) {
      nav_msgs::msg::Odometry msg;
      msg.header.stamp = rclcpp::Time(static_cast<int64_t>(header.stamp), RCL_ROS_TIME);
      msg.header.frame_id = header.frame;
      msg.child_frame_id = body_frame_;
      transformToMsg(odom.pose, msg.pose.pose);
      vecToMsg(odom.linear_velocity, msg.twist.twist.linear);
      vecToMsg(odom.angular_velocity, msg.twist.twist.angular);
      if (!debug_odom_pub_) {
        debug_odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("odom_sensor",
                                                                           rclcpp::QoS(10));
      }
      debug_odom_pub_->publish(msg);
    }
  }

  // Publishes T_world_base (or the native body pose when no base_frame is
  // configured) as odometry + TF at `stamp_ns`.
  void publishState(uint64_t stamp_ns) {
    // Never publish a non-finite state: a NaN TF poisons downstream buffers
    // (tf2 rejects it loudly per message) and a NaN odometry breaks Nav2.
    if (!T_W_I_.matrix().allFinite() || !v_W_.allFinite() || !gyro_I_.allFinite()) {
      LOG_TIMED(E, 5.0, "[Output] estimator state is non-finite; not publishing odometry/TF.");
      return;
    }
    const rclcpp::Time stamp(static_cast<int64_t>(stamp_ns), RCL_ROS_TIME);

    Transform T_W_B = T_W_I_;
    std::string child = body_frame_;
    V3 v_child = T_W_I_.rotation().transpose() * v_W_;
    V3 w_child = gyro_I_;

    if (!frames_.base_frame.empty()) {
      Transform T_L_B;
      if (!lookupSensorBase(stamp, T_L_B)) {
        LOG_TIMED(W, 5.0, "[Output] TF '" << frames_.sensor_frame << "' <- '" << frames_.base_frame
                                          << "' unavailable; skipping odometry publish.");
        return;
      }
      const Transform T_I_B(T_I_L_ * T_L_B);
      T_W_B = Transform(T_W_I_ * T_I_B);
      child = frames_.base_frame;
      // Twist of the base origin, expressed in the base frame. The base origin
      // sits at r = t_I_B in the IMU frame, so its velocity picks up an
      // omega x r term (rigid-body instantaneous velocity; relative articulated
      // motion between TF samples is not observable here and is neglected).
      const M3 R_B_I = T_I_B.rotation().transpose();
      const V3 v_I = T_W_I_.rotation().transpose() * v_W_;  // IMU-frame velocity
      v_child = R_B_I * (v_I + gyro_I_.cross(T_I_B.translation()));
      w_child = R_B_I * gyro_I_;
    }

    // Latch the odom anchor on the first published state: yaw + translation of
    // the initial child (base) pose, so the child starts at the odom origin
    // and odom stays gravity-aligned (roll/pitch of the
    // start pose are NOT folded in). Constant for the whole run.
    if (!anchored_) {
      const M3 R = T_W_B.rotation();
      const double yaw = std::atan2(R(1, 0), R(0, 0));
      const Transform T_anchor(Eigen::AngleAxisd(yaw, V3::UnitZ()).toRotationMatrix(),
                               T_W_B.translation());
      T_anchor_inv_ = Transform(T_anchor.inverse());
      anchored_ = true;
      LOG(I, "[Output] odom anchored at the initial '"
                 << child << "' pose (native-world t = [" << T_W_B.translation().transpose()
                 << "], yaw = " << yaw << " rad).");
    }
    T_W_B = Transform(T_anchor_inv_ * T_W_B);
    // Body-frame twists are invariant under the constant world re-anchor.

    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = world_frame_;
    msg.child_frame_id = child;
    transformToMsg(T_W_B, msg.pose.pose);
    vecToMsg(v_child, msg.twist.twist.linear);
    vecToMsg(w_child, msg.twist.twist.angular);
    odom_pub_->publish(msg);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header = msg.header;
    tf_msg.child_frame_id = child;
    transformToMsg(T_W_B, tf_msg.transform);
    tf_broadcaster_->sendTransform(tf_msg);
  }

  // World-frame clouds are re-expressed through the odom anchor (consistent
  // with the published TF) and forwarded to the native publisher; clouds in
  // other frames (body-frame debug clouds) pass through untouched. The
  // registered scan additionally feeds the accumulated map cloud.
  template <typename CloudT>
  void onCloud(const CloudT& cloud, const Header& header, const std::string& topic,
               const std::string& child_frame) {
    auto native = native_.lock();
    if (!native) {
      return;
    }
    if (header.frame != world_frame_) {
      native->publish(cloud, header, topic, child_frame);
      return;
    }
    // World-frame cloud before the anchor exists (first frames while the
    // base TF is still unresolved): drop it rather than publish data that
    // would later disagree with the anchored TF.
    if (!anchored_) {
      return;
    }
    CloudT anchored = cloud;
    anchored.points() =
        ((T_anchor_inv_.linear() * anchored.points()).colwise() + T_anchor_inv_.translation())
            .eval();
    native->publish(anchored, header, topic, child_frame);

    if (topic == "points/registered" && map_cloud_period_ns_ > 0) {
      accumulateMapCloud(anchored, header);
    }
  }

  // Rolling voxel-downsampled accumulation of the registered scans, published
  // as the `map` topic for RViz/debugging (BIEVR has no native map cloud).
  template <typename CloudT>
  void accumulateMapCloud(const CloudT& registered, const Header& header) {
    Pointcloud scan_down;
    voxelDownsample(Pointcloud(registered), scan_down, map_cloud_voxel_);
    map_cloud_ = map_cloud_ + scan_down;
    // Re-voxelize periodically so overlapping scans do not grow the cloud
    // unboundedly while standing still.
    if (++map_cloud_scans_since_compact_ >= 20) {
      Pointcloud compacted;
      voxelDownsample(map_cloud_, compacted, map_cloud_voxel_);
      map_cloud_ = std::move(compacted);
      map_cloud_scans_since_compact_ = 0;
    }

    if (header.stamp >= next_map_pub_ns_) {
      if (auto native = native_.lock()) {
        native->publish(map_cloud_, header, "map");
      }
      next_map_pub_ns_ = header.stamp + map_cloud_period_ns_;
    }
  }

  // TF(sensor_frame <- base_frame) ladder: exact -> latest -> cached.
  bool lookupSensorBase(const rclcpp::Time& stamp, Transform& T_L_B) {
    geometry_msgs::msg::TransformStamped tf_msg;
    bool resolved = false;
    try {
      tf_msg = tf_buffer_->lookupTransform(frames_.sensor_frame, frames_.base_frame, stamp);
      resolved = true;
    } catch (const tf2::TransformException&) {
      try {
        tf_msg = tf_buffer_->lookupTransform(frames_.sensor_frame, frames_.base_frame,
                                             tf2::TimePointZero);
        resolved = true;
      } catch (const tf2::TransformException&) {
      }
    }
    if (resolved) {
      const auto& q = tf_msg.transform.rotation;
      const auto& t = tf_msg.transform.translation;
      cached_T_L_B_ = Transform(Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix(),
                                V3(t.x, t.y, t.z));
      cached_stamp_ = stamp;
      has_cached_ = true;
      T_L_B = cached_T_L_B_;
      return true;
    }
    if (has_cached_ &&
        std::abs((stamp - cached_stamp_).seconds()) * 1000.0 <= frames_.max_tf_age_ms) {
      T_L_B = cached_T_L_B_;
      return true;
    }
    return false;
  }

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  FramesConfig frames_;
  std::string world_frame_;
  std::string body_frame_;
  Transform T_I_L_;
  uint64_t period_ns_;
  uint64_t map_cloud_period_ns_;
  double map_cloud_voxel_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr debug_odom_pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::weak_ptr<Publisher> native_;

  // Latest (scan-corrected, then IMU-propagated) state.
  bool have_state_ = false;
  Transform T_W_I_;
  V3 v_W_ = V3::Zero();
  V3 gyro_I_ = V3::Zero();
  V3 gyro_bias_ = V3::Zero();
  uint64_t prop_stamp_ = 0;
  uint64_t next_pub_ns_ = 0;

  // Constant odom anchor (latched at the first published state).
  bool anchored_ = false;
  Transform T_anchor_inv_;

  // Accumulated map cloud (debug/RViz).
  Pointcloud map_cloud_;
  int map_cloud_scans_since_compact_ = 0;
  uint64_t next_map_pub_ns_ = 0;

  // Cached TF(sensor <- base) fallback.
  Transform cached_T_L_B_;
  rclcpp::Time cached_stamp_{0, 0, RCL_ROS_TIME};
  bool has_cached_ = false;
};

}  // namespace bievr

#endif  // BIEVR_LIO_ROS2_OUTPUT_MANAGER_H_
