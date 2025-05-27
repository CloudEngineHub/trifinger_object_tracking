/**
 * @file
 * @brief Wrapper on the Pylon Driver to synchronise three pylon dependent
 * cameras.
 * @copyright 2020, Max Planck Gesellschaft. All rights reserved.
 * @license BSD 3-clause
 */
#include <trifinger_object_tracking/tricamera_object_tracking_driver.hpp>

#include <cmath>
#include <cstdlib>
#include <thread>

#include <trifinger_cameras/parse_yml.h>

namespace trifinger_object_tracking
{
// this needs to be declared here...
constexpr std::chrono::milliseconds TriCameraObjectTrackerDriver::rate;

TriCameraObjectTrackerDriver::TriCameraObjectTrackerDriver(
    const std::string& device_id_1,
    const std::string& device_id_2,
    const std::string& device_id_3,
    BaseCuboidModel::ConstPtr cube_model,
    bool downsample_images,
    trifinger_cameras::Settings settings)
    : camera_driver_(
          device_id_1, device_id_2, device_id_3, downsample_images, settings),
      cube_detector_(
          trifinger_object_tracking::create_trifingerpro_cube_detector(
              cube_model))
{
    init_fope();
}

TriCameraObjectTrackerDriver::TriCameraObjectTrackerDriver(
    const std::filesystem::path& camera_calibration_file_1,
    const std::filesystem::path& camera_calibration_file_2,
    const std::filesystem::path& camera_calibration_file_3,
    BaseCuboidModel::ConstPtr cube_model,
    bool downsample_images,
    trifinger_cameras::Settings settings)
    : camera_driver_(camera_calibration_file_1,
                     camera_calibration_file_2,
                     camera_calibration_file_3,
                     downsample_images,
                     settings),
      cube_detector_(
          trifinger_object_tracking::create_trifingerpro_cube_detector(
              cube_model))
{
    init_fope();
}

void TriCameraObjectTrackerDriver::init_fope()
{
    if (const char* fope_config_path = std::getenv("FOPE_CONFIG"))
    {
        fope_ = std::make_unique<fope::PoseEstimator>(
            fope::PoseEstimator::create_from_config(fope_config_path));
    }
}

trifinger_cameras::TriCameraInfo TriCameraObjectTrackerDriver::get_sensor_info()
{
    return camera_driver_.get_sensor_info();
}

TriCameraObjectObservation TriCameraObjectTrackerDriver::get_observation_color()
{
    std::array<cv::Mat, N_CAMERAS> images_bgr;

    TriCameraObjectObservation observation = camera_driver_.get_observation();

    for (size_t i = 0; i < N_CAMERAS; i++)
    {
        cv::cvtColor(
            observation.cameras[i].image, images_bgr[i], cv::COLOR_BayerBG2BGR);
    }

    observation.object_pose =
        cube_detector_.detect_cube_single_thread(images_bgr);

    constexpr float FILTER_CONFIDENCE_THRESHOLD = 0.75;
    constexpr float FILTER_CONFIDENCE_DEGRADATION = 0.9;
    if (previous_pose_.confidence > 0 &&
        observation.object_pose.confidence < FILTER_CONFIDENCE_THRESHOLD)
    {
        // every time a pose is reused, degrade its confidence a bit
        previous_pose_.confidence *= FILTER_CONFIDENCE_DEGRADATION;
        observation.filtered_object_pose = previous_pose_;
    }
    else
    {
        observation.filtered_object_pose = observation.object_pose;
        previous_pose_ = observation.object_pose;
    }

    return observation;
}

TriCameraObjectObservation TriCameraObjectTrackerDriver::get_observation_fope()
{
    std::vector<cv::Mat> images_bgr(N_CAMERAS);

    TriCameraObjectObservation observation = camera_driver_.get_observation();

    for (size_t i = 0; i < N_CAMERAS; i++)
    {
        cv::cvtColor(observation.cameras[i].image,
                     images_bgr[i],
                     cv::COLOR_BayerRGGB2BGR_EA);
    }

    std::vector<std::optional<fope::Pose>> poses =
        fope_->estimate_poses(images_bgr);
    std::optional<fope::Pose> pose = poses.at(0);

    if (pose)
    {
        Eigen::Matrix4d matrix;
        cv::cv2eigen(pose->pose.matrix, matrix);
        observation.object_pose = ObjectPose(matrix);
    }

    // No filtering happening here at the moment
    observation.filtered_object_pose = observation.object_pose;
    previous_pose_ = observation.object_pose;

    return observation;
}

TriCameraObjectObservation TriCameraObjectTrackerDriver::get_observation()
{
    if (fope_)
    {
        return get_observation_fope();
    }
    else
    {
        return get_observation_color();
    }
}

cv::Mat TriCameraObjectTrackerDriver::get_debug_image(bool fill_faces)
{
    if (fope_)
    {
        std::vector<cv::Mat> images_bgr(N_CAMERAS);
        TriCameraObjectObservation observation =
            camera_driver_.get_observation();

        for (size_t i = 0; i < N_CAMERAS; i++)
        {
            cv::cvtColor(observation.cameras[i].image,
                         images_bgr[i],
                         cv::COLOR_BayerRGGB2BGR_EA);
        }

        return fope_->visualize_last_poses(images_bgr);
    }
    else
    {
        get_observation();
        return cube_detector_.create_debug_image(fill_faces);
    }
}

}  // namespace trifinger_object_tracking
