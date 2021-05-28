#include <vector>
#include <opencv2/opencv.hpp>

#include "pose.hpp"
#include "poseEstimation.hpp"

int main(){

	poseEstimation::poseEstimation pe = poseEstimation::poseEstimation("../models/poseEstimationModel.onnx");

	cv::Mat img = cv::imread("../input.jpg");
	std::vector<poseEstimation::Pose> poses = pe.run(img);
	for(int i = 0; i < poses.size(); i++)
		poses[i].draw(img);
	cv::imwrite("../output.jpg", img);

	return 0;
}
