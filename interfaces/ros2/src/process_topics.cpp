#include <bievr_lio/common.h>
#include <bievr_lio/log++.h>
#include <bievr_lio/synchronizer.h>
#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "bievr_lio/config_loader.h"
#include "bievr_lio_ros2/output_manager.h"
#include "bievr_lio_ros2/publisher.h"
#include "bievr_lio_ros2/self_filter.h"
#include "bievr_ros_common/conversions.h"
#ifdef BIEVR_WITH_LIVOX
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif

using namespace std::chrono_literals;

int main(int argc, char** argv) {
  srand(1);
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("bievr_lio_topic_node");

  bievr::Config config;
  // rclcpp::init does not strip ROS arguments from argv; remove_ros_arguments
  // returns just the application arguments (our config-file flags).
  if (!bievr::loadConfigFromArgs(rclcpp::remove_ros_arguments(argc, argv), config)) {
    LOG(E, "Failed to load config.");
    return -1;
  }

  // Cap TBB parallelism for the whole process (0 = TBB default, i.e. all cores).
  const int n_threads =
      config.max_num_threads > 0 ? config.max_num_threads : tbb::this_task_arena::max_concurrency();
  tbb::global_control tbb_control(tbb::global_control::max_allowed_parallelism, n_threads);
  LOG(I, config.max_num_threads > 0, "TBB parallelism limited to " << n_threads << " threads.");

  auto pipeline = std::make_shared<bievr::Pipeline>(config.pipeline_config);
  auto synchronizer = std::make_shared<bievr::Synchronizer>(pipeline);
  auto lio_pub = std::make_shared<bievr::Publisher>(node, pipeline, "bievr_lio");

  // TF buffer: needed by the box self-filter and the base-frame output stage.
  std::shared_ptr<tf2_ros::Buffer> tf_buffer;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener;
  if (config.self_filter.enable || !config.frames.base_frame.empty()) {
    tf_buffer = std::make_shared<tf2_ros::Buffer>(
        node->get_clock(), tf2::durationFromSec(config.frames.tf_buffer_seconds));
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer, node);
  }

  // Self-filter (virtual boxes anchored to TF frames). Runs on the converted
  // cloud before it enters the synchronizer/pipeline.
  std::shared_ptr<bievr::SelfFilter> self_filter;
  if (config.self_filter.enable) {
    if (config.frames.sensor_frame.empty()) {
      LOG(E, "self_filter.enable requires frames.sensor_frame (the TF frame of the raw points).");
      return -1;
    }
    self_filter = std::make_shared<bievr::SelfFilter>(node, tf_buffer, config.self_filter,
                                                      config.frames.sensor_frame);
  }

  // Output stage: base-frame re-expression + high-rate IMU-propagated odometry.
  // Registered after the native Publisher so it takes over Odometry publishing.
  std::shared_ptr<bievr::OutputManager> output_manager;
  if (!config.frames.base_frame.empty() || config.output.odom_rate_hz > 0.0) {
    output_manager = std::make_shared<bievr::OutputManager>(node, tf_buffer, config);
    output_manager->registerWith(pipeline, lio_pub);
  }

  // Input filter (Livox tag/intensity), applied during message conversion.
  const std::array<uint8_t, 256> tag_keep = bievr::makeTagKeepLut(config.input_filter.tag_filter);
  bievr::CloudFilterOptions cloud_filter;
  cloud_filter.tag_keep = config.input_filter.tag_filter.enable ? &tag_keep : nullptr;
  cloud_filter.min_intensity = config.input_filter.min_intensity;

  // ROS2 has no ShapeShifter: discover the pointcloud topic's type from the
  // graph, then use a generic (serialized) subscription to handle whichever of
  // the supported message types is being published on that single topic.
  const std::string& pc_topic = config.topic_config.pointcloud_topic;
  std::string pc_type;
  LOG(I, "Waiting to discover type of pointcloud topic '" << pc_topic << "'...");
  while (rclcpp::ok() && pc_type.empty()) {
    const auto names_types = node->get_topic_names_and_types();
    const auto it = names_types.find(pc_topic);
    if (it != names_types.end() && !it->second.empty()) {
      pc_type = it->second.front();
    } else {
      rclcpp::spin_some(node);
      rclcpp::sleep_for(100ms);
    }
  }
  if (!rclcpp::ok()) {
    rclcpp::shutdown();
    return 0;
  }
  LOG(I, "Pointcloud topic type: " << pc_type);

  // QoS must be compatible with the publisher. SensorDataQoS (best-effort)
  // matches most LiDAR drivers (e.g. Ouster); some drivers (e.g. Livox) publish
  // reliable — adjust here if you see no messages arriving.
  const rclcpp::QoS sensor_qos = rclcpp::SensorDataQoS();

  auto pc_sub = node->create_generic_subscription(
      pc_topic, pc_type, sensor_qos,
      [&, pc_type](std::shared_ptr<rclcpp::SerializedMessage> serialized) {
        if (pc_type == "sensor_msgs/msg/PointCloud2") {
          sensor_msgs::msg::PointCloud2 msg;
          rclcpp::Serialization<sensor_msgs::msg::PointCloud2>().deserialize_message(
              serialized.get(), &msg);
          bievr::StampedIntensityPointcloud pointcloud;
          if (bievr::msgToPointcloud(msg, pointcloud, cloud_filter)) {
            if (self_filter) self_filter->apply(pointcloud);
            synchronizer->addPointcloud(pointcloud);
          }
        }
#ifdef BIEVR_WITH_LIVOX
        else if (pc_type == "livox_ros_driver2/msg/CustomMsg") {
          livox_ros_driver2::msg::CustomMsg msg;
          rclcpp::Serialization<livox_ros_driver2::msg::CustomMsg>().deserialize_message(
              serialized.get(), &msg);
          bievr::StampedIntensityPointcloud pointcloud;
          if (bievr::msgToPointcloud(msg, pointcloud, cloud_filter)) {
            if (self_filter) self_filter->apply(pointcloud);
            synchronizer->addPointcloud(pointcloud);
          }
        }
#endif
        else {
          LOG_FIRST(W, 1, "Received unsupported pointcloud message type: " << pc_type);
        }
      });

  auto imu_sub = node->create_subscription<sensor_msgs::msg::Imu>(
      config.topic_config.imu_topic, sensor_qos, [&](const sensor_msgs::msg::Imu::SharedPtr msg) {
        bievr::ImuMeasurement imu;
        if (!bievr::msgToImuMeasurement(*msg, imu)) {
          return;
        }
        synchronizer->addImu(imu);
        // After the synchronizer: a scan completed by this sample corrects the
        // state before the high-rate propagation publish.
        if (output_manager) output_manager->onImu(imu);
      });

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
