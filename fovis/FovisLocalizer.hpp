//standard include
#include <math.h>

//opencv include
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

//fovis include
#include "mediain.hpp"

#include <fovis.hpp>

class FovisLocalizer {

public:

	FovisLocalizer(const CameraParams &input_params,
			       const cv::Mat& initial_frame);
	~FovisLocalizer();

	void processFrame(const cv::Mat& img, const cv::Mat& depth);
	std::pair<cv::Vec3f,cv::Vec3f> getTransform() const { return _transform; }
	Eigen::Isometry3d transform_eigen() const { return _transform_eigen; }
	void reloadFovis();

	int fv_param_max_pyr_level = 3;
	int fv_param_feature_search_window = 25;
	int fv_param_feature_window_size = 9; //fovis parameters
	int fv_param_target_ppf = 250;

	int num_optical_flow_sectors_x = 4;
	int num_optical_flow_sectors_y = 3; //optical flow parameters
	int num_optical_flow_points = 200;
	int flow_arbitrary_outlier_threshold_int = 200;

private:

	std::pair<cv::Vec3f,cv::Vec3f> _transform;
	Eigen::Isometry3d _transform_eigen;

	fovis::CameraIntrinsicsParameters _rgb_params;
	fovis::Rectification* _rect;
	fovis::VisualOdometry* _odom;

	cv::Mat frameGray, prevGray, depthFrame;

};
