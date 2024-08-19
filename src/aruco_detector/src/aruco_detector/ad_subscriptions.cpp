/**
 * Aruco Detector topic subscription callbacks.
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

#include <aruco_detector/aruco_detector.hpp>

namespace ArucoDetector
{

/**
 * @brief Parses a new image message.
 *
 * @param msg Image message to parse.
 */
void ArucoDetectorNode::camera_callback(
  const Image::ConstSharedPtr & msg,
  const CameraInfo::ConstSharedPtr & camera_info_msg)
{
  // Get camera parameters
  if (get_calibration_params_) {
    cameraMatrix = cv::Mat(3, 3, cv::DataType<double>::type);
    distCoeffs = cv::Mat(1, 5, cv::DataType<double>::type);

    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        cameraMatrix.at<double>(i, j) = camera_info_msg->k[i * 3 + j];
      }
    }

    for (size_t i = 0; i < 5; i++) {
      distCoeffs.at<double>(0, i) = camera_info_msg->d[i];
    }

    // Set coordinate system
    objPoints = cv::Mat(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-aruco_side_ / 2.f, aruco_side_ / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(aruco_side_ / 2.f, aruco_side_ / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(aruco_side_ / 2.f, -aruco_side_ / 2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-aruco_side_ / 2.f, -aruco_side_ / 2.f, 0);

    get_calibration_params_ = false;
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

} // namespace ArucoDetector
