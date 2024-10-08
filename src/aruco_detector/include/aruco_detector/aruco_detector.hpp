/**
 * Aruco Detector node definition.
 *
 * Lorenzo Bianchi <lnz.bnc@gmail.com>
 * dotX Automation s.r.l. <info@dotxautomation.com>
 *
 * August 19, 2024
 */

/**
 * Copyright 2024 dotX Automation s.r.l.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ARUCO_DETECTOR__ARUCO_DETECTOR_HPP
#define ARUCO_DETECTOR__ARUCO_DETECTOR_HPP

#include <algorithm>
#include <atomic>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

#include <semaphore.h>

#include <dua_node/dua_node.hpp>
#include <dua_qos_cpp/dua_qos.hpp>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <rclcpp/rclcpp.hpp>

#include <image_transport/image_transport.hpp>

#include <dua_interfaces/msg/visual_targets.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <std_srvs/srv/set_bool.hpp>

#include <theora_wrappers/publisher.hpp>

using namespace dua_interfaces::msg;
using namespace geometry_msgs::msg;
using namespace rcl_interfaces::msg;
using namespace sensor_msgs::msg;
using namespace std_msgs::msg;
using namespace vision_msgs::msg;

using namespace std_srvs::srv;

#define UNUSED(arg) (void)(arg)
#define LINE std::cout << __FUNCTION__ << ", LINE: " << __LINE__ << std::endl;

namespace aruco_detector
{

/**
 * ArUco marker detection node.
 */
class ArucoDetector : public dua_node::NodeBase
{
public:
  ArucoDetector(const rclcpp::NodeOptions & node_options = rclcpp::NodeOptions());
  ~ArucoDetector();

private:
  /* Node initialization routines. */
  void init_parameters();
  void init_publishers();
  void init_services();
  void init_subscriptions();

  /* image_transport subscriptions. */
  std::shared_ptr<image_transport::CameraSubscriber> camera_sub_;

  /* Topic subscriptions callbacks. */
  void camera_callback(
    const Image::ConstSharedPtr & msg,
    const CameraInfo::ConstSharedPtr & camera_info_msg);

  /* Topic publishers. */
  rclcpp::Publisher<Detection2DArray>::SharedPtr detections_pub_;
  rclcpp::Publisher<VisualTargets>::SharedPtr visual_targets_pub_;

  /* Theora stream publishers. */
  std::shared_ptr<TheoraWrappers::Publisher> stream_pub_;

  /* Service servers. */
  rclcpp::Service<SetBool>::SharedPtr enable_server_;

  /* Service callbacks. */
  void enable_callback(
    SetBool::Request::SharedPtr req,
    SetBool::Response::SharedPtr resp);

  /* Data buffers. */
  cv::Mat new_frame_;
  Header last_header_;

  /* Internal state variables. */
  bool got_camera_info_ = false;
  cv::Mat camera_matrix_, dist_coeffs_, obj_points_;

  /* Node parameters. */
  bool always_publish_stream_ = false;
  double aruco_side_ = 0.0;
  cv::aruco::PredefinedDictionaryType dictionary_ = cv::aruco::DICT_ARUCO_ORIGINAL;
  std::vector<int64_t> valid_ids_;
  int64_t worker_cpu_ = 0;

  /* Synchronization primitives for internal update operations. */
  std::atomic<bool> running_{false};
  sem_t sem1_, sem2_;

  /* Threads. */
  std::thread worker_;

  /* Utility routines. */
  void activate_detector();
  void deactivate_detector();
  Image::SharedPtr frame_to_msg(cv::Mat & frame);
  void publish_frame(cv::Mat & frame, Header & header);
  void rodr_to_quat(cv::Vec3d r, PoseWithCovariance & target_pose);
  bool validate_dictionary(const rclcpp::Parameter & p);
  void worker_thread_routine();
};

} // namespace aruco_detector

#endif // ARUCO_DETECTOR__ARUCO_DETECTOR_HPP
