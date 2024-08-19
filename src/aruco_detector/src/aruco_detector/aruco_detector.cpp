/**
 * Aruco Detector implementation.
 *
 * dotX Automation <info@dotxautomation.com>
 *
 * August 7, 2023
 */

/**
 * This is free software.
 * You can redistribute it and/or modify this file under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 3 of the License, or (at your option) any later
 * version.
 *
 * This file is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this file; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <string>
#include <stdexcept>

#include <aruco_detector/aruco_detector.hpp>

namespace ArucoDetector
{

/**
 * @brief Builds a new Aruco Detector node.
 *
 * @param opts Node options.
 *
 * @throws RuntimeError if initialization fails.
 */
ArucoDetectorNode::ArucoDetectorNode(const rclcpp::NodeOptions & node_options)
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
ArucoDetectorNode::~ArucoDetectorNode()
{
  if (running_.load(std::memory_order_acquire)) {
    // Join worker thread
    running_.store(false, std::memory_order_release);
    sem_post(&sem1_);
    sem_post(&sem2_);
    worker_.join();

    // Shut down camera subscriber
    camera_sub_->shutdown();
    camera_sub_.reset();

    // Destroy semaphores
    sem_destroy(&sem1_);
    sem_destroy(&sem2_);
  }
  stream_pub_.reset();
}

/**
 * @brief Routine to initialize topic subscriptions.
 */
void ArucoDetectorNode::init_subscriptions()
{
  if (autostart_) {
    // Initialize semaphores
    sem_init(&sem1_, 0, 1);
    sem_init(&sem2_, 0, 0);

    // Spawn worker thread
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(
      &ArucoDetectorNode::worker_thread_routine,
      this);
    if (worker_cpu_ != -1) {
      cpu_set_t worker_cpu_set;
      CPU_ZERO(&worker_cpu_set);
      CPU_SET(worker_cpu_, &worker_cpu_set);
      if (pthread_setaffinity_np(
          worker_.native_handle(),
          sizeof(cpu_set_t),
          &worker_cpu_set))
      {
        char err_msg_buf[100] = {};
        char * err_msg = strerror_r(errno, err_msg_buf, 100);
        throw std::runtime_error(
                "ArucoDetectorNode::init_subscriptions: Failed to configure worker thread: " +
                std::string(err_msg));
      }
    }

    // Subscribe to image topic
    camera_sub_ = std::make_shared<image_transport::CameraSubscriber>(
      image_transport::create_camera_subscription(
        this,
        "/image",
        std::bind(
          &ArucoDetectorNode::camera_callback,
          this,
          std::placeholders::_1,
          std::placeholders::_2),
        transport_,
        best_effort_sub_qos_ ?
        dua_qos::BestEffort::get_image_qos(image_sub_depth_).get_rmw_qos_profile() :
        dua_qos::Reliable::get_image_qos(image_sub_depth_).get_rmw_qos_profile()));
  }
}

/**
 * @brief Routine to initialize topic publishers.
 */
void ArucoDetectorNode::init_publishers()
{
  // Targets data
  detections_pub_ = this->create_publisher<Detection2DArray>(
    "/detections",
    dua_qos::Reliable::get_datum_qos());

  // Detections stream
  stream_pub_ = std::make_shared<TheoraWrappers::Publisher>(
    this,
    "/detections_stream",
    dua_qos::BestEffort::get_image_qos(image_sub_depth_).get_rmw_qos_profile());
}

/**
 * @brief Routine to initialize service servers.
 */
void ArucoDetectorNode::init_services()
{
  // Enable
  enable_server_ = this->create_service<SetBool>(
    "~/enable",
    std::bind(
      &ArucoDetectorNode::enable_callback,
      this,
      std::placeholders::_1,
      std::placeholders::_2));
}

/**
 * @brief Worker routine.
 */
void ArucoDetectorNode::worker_thread_routine()
{
  while (true) {
    // Get new data
    std_msgs::msg::Header header_;
    cv::Mat image_{};
    sem_wait(&sem2_);
    if (!running_.load(std::memory_order_acquire)) {
      break;
    }
    image_ = new_frame_.clone();
    header_ = last_header_;
    sem_post(&sem1_);

    // Detect targets
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(
      cv::aruco::DICT_ARUCO_ORIGINAL);

    // Set detector parameters
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);
    detector.detectMarkers(image_, markerCorners, markerIds);

    // Return if no target is detected
    if (markerIds.size() == 0) {continue;}

    std::vector<cv::Vec3d> rvecs(markerIds.size()), tvecs(markerIds.size());

    int n_valid_arucos = 0;

    // Publish information about detected targets
    Detection2DArray detections_msg;
    detections_msg.set__header(header_);
    for (int k = 0; k < int(markerIds.size()); k++) {
      // Continue if target is not valid
      if (std::find(valid_ids_.begin(), valid_ids_.end(), markerIds[k]) == valid_ids_.end())
        continue;

      n_valid_arucos++;

      // Calculate pose for each valid marker
      solvePnP(objPoints, markerCorners[k], cameraMatrix, distCoeffs, rvecs[k], tvecs[k]);

      // Prepare messages to be published
      // Detection message
      Detection2D detection_msg;
      detection_msg.set__id(std::to_string(markerIds[k]));

      ObjectHypothesisWithPose result{};
      result.hypothesis.set__class_id(std::to_string(markerIds[k]));
      result.pose.pose.position.set__x(tvecs[k][0]);
      result.pose.pose.position.set__y(tvecs[k][1]);
      result.pose.pose.position.set__z(tvecs[k][2]);
      rodr_to_quat(rvecs[k], result.pose);

      detection_msg.results.push_back(result);

      detections_msg.detections.push_back(detection_msg);
    }

    if (n_valid_arucos > 0) {
      detections_pub_->publish(detections_msg);

      // Draw search output, ROI and HUD in another image
      cv::aruco::drawDetectedMarkers(image_, markerCorners, markerIds);

      // Draw axis for each marker
      for (int i = 0; i < int(markerIds.size()); i++) {
        cv::drawFrameAxes(image_, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
      }

      camera_frame_ = image_; // Doesn't copy image data, but sets data type...

      // Create processed image message
      Image::SharedPtr processed_image_msg = frame_to_msg(camera_frame_);
      processed_image_msg->set__header(header_);

      // Publish processed image
      stream_pub_->publish(processed_image_msg);
    }
  }

  RCLCPP_WARN(this->get_logger(), "Aruco Detector DEACTIVATED");
}

} // namespace ArucoDetector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ArucoDetector::ArucoDetectorNode)
