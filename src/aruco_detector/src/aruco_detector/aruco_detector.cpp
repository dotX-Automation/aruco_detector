/**
 * Aruco Detector implementation.
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

#include <string>

#include <aruco_detector/aruco_detector.hpp>

namespace aruco_detector
{

/**
 * @brief Builds a new Aruco Detector node.
 *
 * @param opts Node options.
 */
ArucoDetector::ArucoDetector(const rclcpp::NodeOptions & node_options)
: NodeBase("aruco_detector", node_options, true)
{
  init_parameters();
  init_publishers();
  init_subscriptions();
  init_services();

  RCLCPP_INFO(this->get_logger(), "Node initialized");
}

/**
 * @brief Finalizes node operation.
 */
ArucoDetector::~ArucoDetector()
{
  if (running_.load(std::memory_order_acquire)) {
    deactivate_detector();
  }
  stream_pub_.reset();
}

/**
 * @brief Routine to initialize topic subscriptions.
 */
void ArucoDetector::init_subscriptions()
{
  if (this->get_parameter("autostart").as_bool()) {
    activate_detector();
  }
}

/**
 * @brief Routine to initialize topic publishers.
 */
void ArucoDetector::init_publishers()
{
  // detections
  detections_pub_ = this->create_publisher<Detection2DArray>(
    "~/detections",
    dua_qos::Reliable::get_datum_qos());

  // detections_stream
  stream_pub_ = std::make_shared<TheoraWrappers::Publisher>(
    this,
    "~/detections_stream",
    dua_qos::BestEffort::get_image_qos().get_rmw_qos_profile());
}

/**
 * @brief Routine to initialize service servers.
 */
void ArucoDetector::init_services()
{
  // enable
  enable_server_ = this->create_service<SetBool>(
    "~/enable",
    std::bind(
      &ArucoDetector::enable_callback,
      this,
      std::placeholders::_1,
      std::placeholders::_2));
}

/**
 * @brief Worker routine.
 */
void ArucoDetector::worker_thread_routine()
{
  // Instantiate detector
  cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictionary_);
  cv::aruco::DetectorParameters detector_params = cv::aruco::DetectorParameters();
  detector_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  cv::aruco::ArucoDetector detector(dictionary, detector_params);

  while (true) {
    // Get new data
    Header header_{};
    cv::Mat image_{};
    sem_wait(&sem2_);
    if (!running_.load(std::memory_order_acquire)) {
      break;
    }
    image_ = new_frame_.clone();
    header_ = last_header_;
    sem_post(&sem1_);

    // Detect targets
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    detector.detectMarkers(image_, marker_corners, marker_ids);

    // Drop sample if no target is detected
    if (marker_ids.size() == 0) {
      continue;
    }

    std::vector<cv::Vec3d> rvecs(marker_ids.size()), tvecs(marker_ids.size());

    int n_valid_arucos = 0;

    // Publish information about detected targets
    Detection2DArray detections_msg;
    detections_msg.set__header(header_);
    for (int k = 0; k < int(marker_ids.size()); k++) {
      // Continue if target is not valid
      if (std::find(valid_ids_.begin(), valid_ids_.end(), marker_ids[k]) == valid_ids_.end()) {
        continue;
      }

      n_valid_arucos++;

      // Calculate pose for each valid marker
      cv::solvePnP(
        obj_points_, marker_corners[k], camera_matrix_, dist_coeffs_, rvecs[k], tvecs[k]);

      // Compute marker center position
      cv::Point2f center(0.0f, 0.0f);
      for (const auto & corner : marker_corners[k]) {
        center += corner;
      }
      center *= 0.25f;

      // Compute marker orientation in the image plane
      cv::Point2f top_left = marker_corners[k][0];
      cv::Point2f top_right = marker_corners[k][1];
      cv::Point2f direction = top_right - top_left;
      double theta = double(cv::fastAtan2(-direction.y, direction.x)) * CV_PI / 180.0;

      // Compute marker size in pixels (making the assumption that it is perfectly square)
      double size = cv::norm(direction);

      // Prepare messages to be published
      // Detection message
      Detection2D detection_msg{};
      detection_msg.set__header(header_);

      // Object hypothesis with pose
      ObjectHypothesisWithPose result{};
      result.hypothesis.set__class_id("ArUco " + std::to_string(marker_ids[k]));
      result.hypothesis.set__score(1.0);
      result.pose.pose.position.set__x(tvecs[k][0]);
      result.pose.pose.position.set__y(tvecs[k][1]);
      result.pose.pose.position.set__z(tvecs[k][2]);
      rodr_to_quat(rvecs[k], result.pose);

      // Bounding box
      BoundingBox2D bbox{};
      bbox.center.position.set__x(center.x);
      bbox.center.position.set__y(center.y);
      bbox.center.set__theta(theta);
      bbox.set__size_x(size);
      bbox.set__size_y(size);
      detection_msg.set__bbox(bbox);

      detection_msg.results.push_back(result);

      detections_msg.detections.push_back(detection_msg);
    }

    if (n_valid_arucos > 0) {
      detections_pub_->publish(detections_msg);

      // Draw search output in another image
      cv::aruco::drawDetectedMarkers(image_, marker_corners, marker_ids);

      // Draw axis for each marker
      for (int i = 0; i < int(marker_ids.size()); i++) {
        cv::drawFrameAxes(image_, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], 0.1);
      }

      camera_frame_ = image_; // Doesn't copy image data, but sets data type...

      // Create processed image message
      Image::SharedPtr processed_image_msg = frame_to_msg(camera_frame_);
      processed_image_msg->set__header(header_);

      // Publish processed image
      stream_pub_->publish(processed_image_msg);
    }
  }

  RCLCPP_WARN(this->get_logger(), "ArUco Detector DEACTIVATED");
}

} // namespace aruco_detector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(aruco_detector::ArucoDetector)
