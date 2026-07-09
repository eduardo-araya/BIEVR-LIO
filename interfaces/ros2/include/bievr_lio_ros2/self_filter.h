#ifndef BIEVR_LIO_ROS2_SELF_FILTER_H_
#define BIEVR_LIO_ROS2_SELF_FILTER_H_

// TF-aware self-filter: removes points of the carrier vehicle's own body from
// every incoming scan before the pipeline sees them. Each configured box is an
// axis-aligned box in its own frame (a URDF link plus a fixed offset), so
// articulated links of the carrier are tracked through TF.
//
// Semantics:
//  * The point-in-box test runs in the BOX-LOCAL frame: each LiDAR-frame point
//    is transformed through TF(box.frame_id <- lidar_frame) at the scan stamp,
//    then through the inverse box offset; it is dropped when |p| <= size/2 on
//    all axes for ANY box.
//  * TF lookup ladder per box: exact stamp -> latest available -> cached value
//    (<= max_tf_age_ms old). The box frames sit past dynamic joints whose /tf
//    lags the scan stamp, so the exact-stamp lookup routinely fails; without
//    the fallbacks the filter silently skips boxes and robot-body artifacts
//    enter the map.
//  * FAIL OPEN: if every box TF lookup fails, or the filter would drop 100% of
//    the scan, the cloud passes through unfiltered (odometry survival beats
//    filter correctness).
//
// Dropped points get their xyz zeroed (range-0), so the pipeline's min-range
// filter removes them; time/intensity rows stay untouched.

#include <tf2_ros/buffer.h>

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>

#include "bievr_lio/common.h"
#include "bievr_lio/config_loader.h"
#include "bievr_lio/log++.h"

namespace bievr {

class SelfFilter {
 public:
  SelfFilter(rclcpp::Node::SharedPtr node, std::shared_ptr<tf2_ros::Buffer> tf_buffer,
             const SelfFilterConfig& config, const std::string& lidar_frame)
      : tf_buffer_(std::move(tf_buffer)), config_(config), lidar_frame_(lidar_frame) {
    boxes_.reserve(config.boxes.size());
    for (const auto& bc : config.boxes) {
      Box box;
      box.config = bc;
      // RPY convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
      box.R_offset = (Eigen::AngleAxisd(bc.offset_rpy.z(), V3::UnitZ()) *
                      Eigen::AngleAxisd(bc.offset_rpy.y(), V3::UnitY()) *
                      Eigen::AngleAxisd(bc.offset_rpy.x(), V3::UnitX()))
                         .toRotationMatrix();
      box.half = bc.size / 2.0;
      boxes_.push_back(box);
    }
    if (config_.visualize) {
      marker_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
          "self_filter/markers", rclcpp::QoS(1).transient_local());
    }
  }

  // Zeroes the xyz of every point inside any box. `cloud_L` is in the LiDAR
  // frame; `cloud_L.stamp` (ns) selects the TF sample.
  void apply(StampedIntensityPointcloud& cloud_L) {
    if (!config_.enable || boxes_.empty() || cloud_L.empty()) {
      return;
    }
    const rclcpp::Time stamp(static_cast<int64_t>(cloud_L.stamp), RCL_ROS_TIME);

    // Resolve TF(box <- lidar) per box; skip boxes with no usable transform.
    std::vector<const Box*> active;
    active.reserve(boxes_.size());
    for (auto& box : boxes_) {
      if (resolveBoxTransform(box, stamp)) {
        active.push_back(&box);
      }
    }
    if (active.empty()) {
      LOG_TIMED(W, 5.0, "[SelfFilter] no box TF available (lidar_frame='"
                            << lidar_frame_ << "'); cloud passed through unfiltered.");
      return;
    }

    auto& data = cloud_L.data();
    const size_t n = cloud_L.size();
    std::vector<char> drop(n, 0);
    size_t n_drop = 0;
    for (size_t i = 0; i < n; ++i) {
      const V3 p_L = data.col(i).head<3>();
      if (p_L.isZero()) continue;  // already filtered upstream
      for (const Box* box : active) {
        // p_box = R_off^T * (R_bl * p_L + t_bl - t_off)
        const V3 p_frame = box->R_frame_lidar * p_L + box->t_frame_lidar;
        const V3 p_box = box->R_offset.transpose() * (p_frame - box->config.offset_translation);
        if ((p_box.array().abs() <= box->half.array()).all()) {
          drop[i] = 1;
          ++n_drop;
          break;
        }
      }
    }

    // Fail open: a filter that would delete the whole scan is misconfigured
    // (or TF is wrong); keep the data.
    if (n_drop == n) {
      LOG_TIMED(W, 5.0, "[SelfFilter] filter would drop ALL " << n
                                                              << " points; passing cloud through.");
      return;
    }
    for (size_t i = 0; i < n; ++i) {
      if (drop[i]) {
        data.col(i).head<3>().setZero();
      }
    }

    total_points_ += n;
    total_dropped_ += n_drop;
    if (++n_clouds_ % 200 == 1) {
      LOG(I, "[SelfFilter] dropped " << total_dropped_ << "/" << total_points_ << " points ("
                                     << (100.0 * static_cast<double>(total_dropped_) /
                                         static_cast<double>(std::max<uint64_t>(total_points_, 1)))
                                     << "%) over " << n_clouds_ << " clouds.");
    }

    maybePublishMarkers(stamp);
  }

 private:
  struct Box {
    SelfFilterBoxConfig config;
    M3 R_offset = M3::Identity();
    V3 half = V3::Zero();
    // Latest resolved TF(box.frame_id <- lidar_frame).
    M3 R_frame_lidar = M3::Identity();
    V3 t_frame_lidar = V3::Zero();
    rclcpp::Time cached_stamp{0, 0, RCL_ROS_TIME};
    bool has_cached = false;
  };

  // Exact stamp -> latest available -> cached (<= max age). Updates the box's
  // cached transform on success; returns false when nothing usable exists.
  bool resolveBoxTransform(Box& box, const rclcpp::Time& stamp) {
    geometry_msgs::msg::TransformStamped tf_msg;
    bool resolved = false;
    try {
      tf_msg = tf_buffer_->lookupTransform(box.config.frame_id, lidar_frame_, stamp);
      resolved = true;
    } catch (const tf2::TransformException&) {
      try {
        tf_msg = tf_buffer_->lookupTransform(box.config.frame_id, lidar_frame_,
                                             tf2::TimePointZero);
        resolved = true;
      } catch (const tf2::TransformException&) {
      }
    }
    if (resolved) {
      const auto& q = tf_msg.transform.rotation;
      const auto& t = tf_msg.transform.translation;
      box.R_frame_lidar = Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix();
      box.t_frame_lidar = V3(t.x, t.y, t.z);
      box.cached_stamp = stamp;
      box.has_cached = true;
      return true;
    }
    // Cached fallback.
    return box.has_cached &&
           std::abs((stamp - box.cached_stamp).seconds()) * 1000.0 <= kMaxCachedAgeMs;
  }

  void maybePublishMarkers(const rclcpp::Time& stamp) {
    if (!marker_pub_ || config_.visualize_rate_hz <= 0.0) {
      return;
    }
    if (last_marker_stamp_.nanoseconds() != 0 &&
        (stamp - last_marker_stamp_).seconds() < 1.0 / config_.visualize_rate_hz) {
      return;
    }
    last_marker_stamp_ = stamp;

    visualization_msgs::msg::MarkerArray markers;
    int id = 0;
    for (const auto& box : boxes_) {
      visualization_msgs::msg::Marker m;
      m.header.stamp = stamp;
      m.header.frame_id = box.config.frame_id;
      m.ns = "self_filter/boxes";
      m.id = id++;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.position.x = box.config.offset_translation.x();
      m.pose.position.y = box.config.offset_translation.y();
      m.pose.position.z = box.config.offset_translation.z();
      const Eigen::Quaterniond q(box.R_offset);
      m.pose.orientation.w = q.w();
      m.pose.orientation.x = q.x();
      m.pose.orientation.y = q.y();
      m.pose.orientation.z = q.z();
      m.scale.x = box.config.size.x();
      m.scale.y = box.config.size.y();
      m.scale.z = box.config.size.z();
      m.color.r = 1.0f;
      m.color.g = 0.3f;
      m.color.b = 0.1f;
      m.color.a = 0.35f;
      markers.markers.push_back(m);

      visualization_msgs::msg::Marker label = m;
      label.ns = "self_filter/labels";
      label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      label.text = box.config.name;
      label.scale.x = label.scale.y = 0.0;
      label.scale.z = 0.4;
      label.color.r = label.color.g = label.color.b = 1.0f;
      label.color.a = 0.9f;
      markers.markers.push_back(label);
    }
    marker_pub_->publish(markers);
  }

  static constexpr double kMaxCachedAgeMs = 500.0;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  SelfFilterConfig config_;
  std::string lidar_frame_;
  std::vector<Box> boxes_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Time last_marker_stamp_{0, 0, RCL_ROS_TIME};

  uint64_t total_points_ = 0;
  uint64_t total_dropped_ = 0;
  uint64_t n_clouds_ = 0;
};

}  // namespace bievr

#endif  // BIEVR_LIO_ROS2_SELF_FILTER_H_
