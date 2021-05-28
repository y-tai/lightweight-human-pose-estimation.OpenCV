#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
namespace poseEstimation{

    struct keypoint{
        cv::Point point;
        float conf;
        int idx;
        keypoint(cv::Point point,float conf,int idx):point(point),conf(conf),idx(idx){}
    };

    class Pose{
    private:
        const std::vector<std::string> kpt_names{"nose", "neck",
                 "r_sho", "r_elb", "r_wri", "l_sho", "l_elb", "l_wri",
                 "r_hip", "r_knee", "r_ank", "l_hip", "l_knee", "l_ank",
                 "r_eye", "l_eye",
                 "r_ear", "l_ear"};
        const std::vector<float> sigmas{0.026     , 0.079     , 0.079     , 0.072     , 0.062     ,
        0.079     , 0.072     , 0.062     , 0.10700001, 0.087     ,
        0.089     , 0.10700001, 0.087     , 0.089     , 0.025     ,
        0.025     , 0.035     , 0.035    };
        const std::vector<float> vars{0.002704  , 0.024964  , 0.024964  , 0.020736  , 0.015376  ,
        0.024964  , 0.020736  , 0.015376  , 0.04579601, 0.030276  ,
        0.031684  , 0.04579601, 0.030276  , 0.031684  , 0.0025    ,
        0.0025    , 0.0049    , 0.0049   };

        const std::vector<std::vector<int>> BODY_PARTS_KPT_IDS = 
        {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11},
                      {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 16}, {5, 17}};


        public:
        std::vector<cv::Point> keypoints;
        float confidence;
        const static int num_kpts = 18;
        Pose(std::vector<cv::Point> keypoints, float confidence=1):keypoints(keypoints),confidence(confidence){
        }

        void draw(cv::Mat img){
            for(int part_id = 0 ;part_id < this->BODY_PARTS_KPT_IDS.size() - 2 ; part_id++){
                int kpt_a_id = this->BODY_PARTS_KPT_IDS[part_id][0];
                int global_kpt_a_id = this->keypoints[kpt_a_id].x;
                if(global_kpt_a_id != -1){
                    cv::Point p_a = this->keypoints[kpt_a_id];
                    cv::circle(img ,p_a ,3, cv::Scalar(0,255,255), -1);
                }
                int kpt_b_id = this->BODY_PARTS_KPT_IDS[part_id][1];
                int global_kpt_b_id = this->keypoints[kpt_b_id].x;
                if(global_kpt_b_id != -1){
                    cv::Point p_b = this->keypoints[kpt_b_id];
                    cv::circle(img ,p_b ,3, cv::Scalar(0,255,255), -1);
                }
                if(global_kpt_a_id != -1 && global_kpt_b_id != -1){
                    cv::Point p_a = this->keypoints[kpt_a_id];
                    cv::Point p_b = this->keypoints[kpt_b_id];
                    cv::line(img ,p_a ,p_b, cv::Scalar(0,255,255), 2);
                }
            }
        }
    };
}