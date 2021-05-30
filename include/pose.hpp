#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
namespace poseEstimation{

    struct keypoint{
        cv::Point point;
        float conf;
        int idx;
        keypoint(cv::Point point,float conf,int idx):
            point(point),conf(conf),idx(idx){}
    };

    class Pose{
    public:
        std::vector<cv::Point> keypoints;
        float confidence;
        cv::Rect bbox;
        int track_id;
        const static int num_kpts = 18;
        Pose(std::vector<cv::Point> keypoints, float confidence=1):
            keypoints(keypoints),confidence(confidence){
                this->bbox = this->get_bbox();                
                this->track_id = -1;
        }

        Pose& operator=(const Pose& obj){
            this->keypoints = obj.keypoints;
            this->confidence = obj.confidence;
            this->bbox = obj.bbox;
            this->track_id = obj.track_id;
            return *this;
        }

        void draw(cv::Mat &img, bool draw_track_id = false){
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
            if(draw_track_id && this->track_id != -1)
                cv::putText(img, std::to_string(this->track_id), cv::Point(this->bbox.x - 20,this->bbox.y + 30),cv::FONT_HERSHEY_PLAIN,3,cv::Scalar(0,0,255),3,8);

        }

    private:
        const std::vector<std::string> kpt_names{"nose", "neck",
                 "r_sho", "r_elb", "r_wri", "l_sho", "l_elb", "l_wri",
                 "r_hip", "r_knee", "r_ank", "l_hip", "l_knee", "l_ank",
                 "r_eye", "l_eye",
                 "r_ear", "l_ear"};

        const std::vector<std::vector<int>> BODY_PARTS_KPT_IDS = 
        {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11},
                      {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 16}, {5, 17}};


        cv::Rect get_bbox(){
            std::vector<cv::Point> found_keypoints(this->keypoints.size());
            auto it = std::copy_if(this->keypoints.begin(), this->keypoints.end(), found_keypoints.begin(),
                [](const cv::Point& p){ return p.x != -1;});
            found_keypoints.resize(std::distance(found_keypoints.begin(),it));
            return cv::boundingRect(found_keypoints);
        }
    };

    class poseTracker{        
    public:
        void track(std::vector<Pose> &current_poses, int threshold = 3){
            std::sort(current_poses.begin(), current_poses.end(),
                [](const Pose& a, const Pose& b){
                    return a.confidence>b.confidence;  
                });
            std::vector<int> mask(previous_poses.size(), 1);
            for(int i = 0; i < current_poses.size(); i++){
                int best_matched_id = -1, best_matched_pose_id = -1, best_matched_iou = -1;
                for(int j = 0; j < this->previous_poses.size(); j++){
                    if(!mask[j])
                        continue;
                    int iou = get_similarity(current_poses[i], this->previous_poses[j]);
                    if(iou > best_matched_iou){
                        best_matched_id = j;
                        best_matched_pose_id = this->previous_poses[j].track_id;
                        best_matched_iou = iou;
                    }
                }
                if(best_matched_iou >= threshold)
                    mask[best_matched_id] = 0;
                else
                    best_matched_pose_id = ++this->max_pose_id;
               current_poses[i].track_id = best_matched_pose_id;
            }
            this->previous_poses = current_poses;
        }

    private:
        int max_pose_id = -1;
        std::vector<Pose> previous_poses;

        const std::vector<float> vars{0.002704  , 0.024964  , 0.024964  , 0.020736  , 0.015376  ,
        0.024964  , 0.020736  , 0.015376  , 0.04579601, 0.030276  ,
        0.031684  , 0.04579601, 0.030276  , 0.031684  , 0.0025    ,
        0.0025    , 0.0049    , 0.0049   };

        int get_similarity(const Pose& a, const Pose& b, float threshold = 0.5){
            int num_similar_kpt = 0;
            for(int kpt_id = 0; kpt_id < Pose::num_kpts; kpt_id++){
                if(a.keypoints[kpt_id].x != -1 && b.keypoints[kpt_id].x != -1){
                    double distance = cv::norm(a.keypoints[kpt_id] - b.keypoints[kpt_id]);
                    double area = std::max(a.bbox.area(), b.bbox.area());
                    double similarity = exp( -distance / (2 * area) * poseTracker::vars[kpt_id]);
                    if(similarity > threshold)
                        num_similar_kpt++;
                }
            }
            return num_similar_kpt;
        }
    };
}