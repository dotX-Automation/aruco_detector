/**
 * Aruco Detector service callbacks.
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
 * @brief Toggles target detection.
 *
 * @param req Service request to parse.
 * @param rest Service response to populate.
 */
void ArucoDetector::enable_callback(
  SetBool::Request::SharedPtr req,
  SetBool::Response::SharedPtr resp)
{
  if (req->data) {
    if (!running_.load(std::memory_order_acquire)) {
      activate_detector();
    }
    resp->set__success(true);
    resp->set__message("");
  } else {
    if (running_.load(std::memory_order_acquire)) {
      deactivate_detector();
    }
    resp->set__success(true);
    resp->set__message("");
  }
}

} // namespace aruco_detector
