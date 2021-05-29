#include <vector>
#include <opencv2/opencv.hpp>

#include "pose.hpp"
#include "poseEstimation.hpp"

int main(){

	poseEstimation::poseEstimation pe("../models/poseEstimationModel.onnx");
	poseEstimation::poseTracker pt;

	
	cv::Mat img = cv::imread("../input.jpg");
	std::vector<poseEstimation::Pose> poses = pe.run(img);
	pt.track(poses);
	for(int i = 0; i < poses.size(); i++)
		poses[i].draw(img, true);
	cv::imwrite("../output.jpg", img);

	return 0;
}
