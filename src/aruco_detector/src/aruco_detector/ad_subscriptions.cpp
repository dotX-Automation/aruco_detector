/**
 * Aruco Detector topic subscription callbacks.
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

#include <aruco_detector/aruco_detector.hpp>

namespace aruco_detector
{

/**
 * @brief Parses a new image message.
 *
 * @param msg Image message to parse.
 * @param camera_info_msg Corresponding camera info message.
 */
void ArucoDetector::camera_callback(
  const Image::ConstSharedPtr & msg,
  const CameraInfo::ConstSharedPtr & camera_info_msg)
{
  // Get camera parameters
  if (!got_camera_info_) {
    camera_matrix_ = cv::Mat(3, 3, cv::DataType<double>::type);
    dist_coeffs_ = cv::Mat(1, 5, cv::DataType<double>::type);

    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        camera_matrix_.at<double>(i, j) = camera_info_msg->k[i * 3 + j];
      }
    }

    for (size_t i = 0; i < 5; i++) {
      dist_coeffs_.at<double>(0, i) = camera_info_msg->d[i];
    }

    // Set coordinate system
    obj_points_ = cv::Mat(4, 1, CV_32FC3);
    obj_points_.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-aruco_side_ / 2.f, aruco_side_ / 2.f, 0);
    obj_points_.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(aruco_side_ / 2.f, aruco_side_ / 2.f, 0);
    obj_points_.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(aruco_side_ / 2.f, -aruco_side_ / 2.f, 0);
    obj_points_.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-aruco_side_ / 2.f, -aruco_side_ / 2.f, 0);

    got_camera_info_ = true;
  }

  // Convert msg to OpenCV image
  cv::Mat frame = cv::Mat(
    msg->height,
    msg->width,
    CV_8UC3,
    (void *)(msg->data.data()));

  // Pass data to worker thread
  sem_wait(&sem1_);
  new_frame_ = frame.clone();
  last_header_ = msg->header;
  sem_post(&sem2_);
}

} // namespace aruco_detector
