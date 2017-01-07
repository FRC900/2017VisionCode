#include <opencv2/core/core.hpp>

class FlowLocalizer 
{

public:
	FlowLocalizer(const cv::Mat &initial_frame);
	void processFrame(const cv::Mat &frame);
	cv::Mat transform_mat() const { return _transform_mat; }
	//cv::Point transform_point(cv::Point input) const { return _transform_mat * input; } 
private:
	cv::Mat _prevFrame;
	cv::Mat _transform_mat;
};
