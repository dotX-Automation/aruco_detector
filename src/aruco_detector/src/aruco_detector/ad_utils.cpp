/**
 * Aruco Detector auxiliary functions.
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

#include <stdexcept>

#include <pthread.h>
#include <sched.h>

#include <aruco_detector/aruco_detector.hpp>

namespace aruco_detector
{

/**
 * @brief Converts a frame into an Image message.
 *
 * @param frame cv::Mat storing the frame.
 * @return Shared pointer to a new Image message.
 */
Image::SharedPtr ArucoDetector::frame_to_msg(cv::Mat & frame)
{
  auto ros_image = std::make_shared<Image>();

  // Set frame-relevant image contents
  ros_image->set__width(frame.cols);
  ros_image->set__height(frame.rows);
  ros_image->set__encoding(sensor_msgs::image_encodings::BGR8);
  ros_image->set__step(frame.cols * frame.elemSize());

  // Check data endianness
  int num = 1;
  ros_image->set__is_bigendian(!(*(char *)&num == 1));

  // Copy frame data (this avoids the obsolete cv_bridge)
  size_t size = ros_image->step * frame.rows;
  ros_image->data.resize(size);
  memcpy(ros_image->data.data(), frame.data, size);

  return ros_image;
}

/**
 * @brief Function to convert Rodrigues' vector to quaternion.
 *
 * @param r Vector in Rodrigues' form (axis-angle).
 * @param target_pose Pose message to fill.
 */
void ArucoDetector::rodr_to_quat(cv::Vec3d r, PoseWithCovariance & target_pose)
{
  double w, x, y, z;
  double angle = cv::norm(r);

  if (angle < 1e-8) {
    w = 1.0;
    x = y = z = 0.0;
  } else {
    double c = std::cos(angle / 2.0);
    double s = std::sin(angle / 2.0);

    w = c;
    x = s * r[0] / angle;
    y = s * r[1] / angle;
    z = s * r[2] / angle;
  }

  target_pose.pose.orientation.set__w(w);
  target_pose.pose.orientation.set__x(x);
  target_pose.pose.orientation.set__y(y);
  target_pose.pose.orientation.set__z(z);
}

/**
 * @brief Activates the detector.
 *
 * @throws std::runtime_error if the worker thread cannot be configured.
 */
void ArucoDetector::activate_detector()
{
  // Initialize semaphores
  sem_init(&sem1_, 0, 1);
  sem_init(&sem2_, 0, 0);

  // Spawn worker thread
  running_.store(true, std::memory_order_release);
  worker_ = std::thread(
    &ArucoDetector::worker_thread_routine,
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
              "ArucoDetector::activate_detector: Failed to configure worker thread: " +
              std::string(err_msg));
    }
  }

  // Subscribe to image topic
  int64_t subscriber_depth = this->get_parameter("subscriber_depth").as_int();
  camera_sub_ = std::make_shared<image_transport::CameraSubscriber>(
    image_transport::create_camera_subscription(
      this,
      this->get_parameter("subscriber_base_topic_name").as_string(),
      std::bind(
        &ArucoDetector::camera_callback,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      this->get_parameter("subscriber_transport").as_string(),
      this->get_parameter("subscriber_best_effort_qos").as_bool() ?
      dua_qos::BestEffort::get_image_qos(subscriber_depth).get_rmw_qos_profile() :
      dua_qos::Reliable::get_image_qos(subscriber_depth).get_rmw_qos_profile()));

  RCLCPP_WARN(this->get_logger(), "ArUco Detector ACTIVATED");
}

/**
 * @brief Deactivates the detector.
 */
void ArucoDetector::deactivate_detector()
{
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

} // namespace aruco_detector
