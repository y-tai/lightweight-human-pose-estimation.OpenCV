#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <ctime>
#include <iostream>
#include "pose.hpp"
namespace poseEstimation{
    class poseEstimation{
    private:
        int in_width = -1;
        int in_height = -1;
        const std::vector<std::vector<int>> BODY_PARTS_PAF_IDS = 
        {{12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25}, {0, 1}, {2, 3}, {4, 5},
        {6, 7}, {8, 9}, {10, 11}, {28, 29}, {30, 31}, {34, 35}, {32, 33}, {36, 37}, {18, 19}, {26, 27}};
        const std::vector<std::vector<int>> BODY_PARTS_KPT_IDS = 
        {{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11},
                      {11, 12}, {12, 13}, {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 16}, {5, 17}};

        cv::dnn::Net net;
        float resize_ratio=1;
        float resize_left_padding = 0;
        float resize_top_padding = 0;

        const int stride = 8;
        const int upsample_ratio = 2;

        int longSize = 480;
    
    public:
        poseEstimation(std::string modelPath){
            this->net = cv::dnn::readNetFromONNX(modelPath);
            this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
    
        std::vector<Pose> run(const cv::Mat& img, const bool multi_person = false){
            std::vector<cv::Mat> outs = this->forward(img);
            std::vector<Pose> res = this->postProcess(outs);
            
            return res;
        }

    private:
        cv::Mat preProcess(const cv::Mat& src){
            float height = src.rows, width = src.cols;
            float ratio = std::min(this->longSize / height ,this->longSize / width);
            int new_height = floor(height * ratio /  this->stride) * this->stride;
            int new_width = floor(width * ratio / this->stride) * this->stride;
            this->in_height = new_height;
            this->in_width = new_width;
                
            cv::Mat dst = this->pad_resize(src, this->in_height, this->in_width);
            return dst;
        }
    
        std::vector<cv::Mat> forward(const cv::Mat& img){
            cv::Mat src = this->preProcess(img);            
            cv::Mat inputBlob = cv::dnn::blobFromImage(src, 1.0 / 255, cv::Size(this->in_width, this->in_height), cv::Scalar(128,128,128), false, false);
    		this->net.setInput(inputBlob);

            std::vector<cv::String> names{"stage_1_output_1_heatmaps", "stage_1_output_0_pafs"};
            std::vector<cv::Mat> outs;
    		net.forward(outs,names);
            return outs;
        }
    
        std::vector<Pose> postProcess(const std::vector<cv::Mat>& outs){

            cv::Mat heatmaps =outs[0];
            std::vector<cv::Mat> heatmaps_channels(heatmaps.size[1] - 1);
            for(int i = 0;i<heatmaps.size[1] - 1;i++){
    		    cv::Mat heatMap(heatmaps.size[2], heatmaps.size[3], CV_32F, reinterpret_cast<float*>(const_cast<uchar*>(heatmaps.ptr(0, i))));
                cv::resize(heatMap,heatMap,cv::Size(0,0),this->upsample_ratio,this->upsample_ratio,cv::INTER_CUBIC);
                heatmaps_channels[i]=heatMap;
            }

            cv::Mat pafs = outs[1]; 
            std::vector<cv::Mat> pafs_channels(pafs.size[1]);
            for(int i = 0;i<pafs.size[1];i++){
    		    cv::Mat paf(heatmaps.size[2], heatmaps.size[3], CV_32F, reinterpret_cast<float*>(const_cast<uchar*>(pafs.ptr(0,i))));
                cv::resize(paf,paf,cv::Size(0,0),this->upsample_ratio,this->upsample_ratio,cv::INTER_CUBIC);
                pafs_channels[i] = paf;
            }

            std::vector<std::vector<keypoint>> all_keypoints_by_type; 
            int total_keypoint_num = 0;

            for(int i = 0; i < heatmaps_channels.size(); i++)
                total_keypoint_num += this->extract_keypoints(heatmaps_channels[i], all_keypoints_by_type, total_keypoint_num); //150

            cv::Mat pose_entries;
            std::vector<keypoint> all_keypoints;
            this->group_keypoints(all_keypoints_by_type, pafs_channels, pose_entries, all_keypoints);

            for(int kpt_id = 0;kpt_id< all_keypoints.size();kpt_id++){
                all_keypoints[kpt_id].point.x*=this->stride/this->upsample_ratio;
                all_keypoints[kpt_id].point.x-=this->resize_left_padding;
                all_keypoints[kpt_id].point.x/=this->resize_ratio;
                all_keypoints[kpt_id].point.y*=this->stride/this->upsample_ratio;                
                all_keypoints[kpt_id].point.y-=this->resize_top_padding;
                all_keypoints[kpt_id].point.y/=this->resize_ratio;
            }

            std::vector<Pose> current_poses;
            for(int i=0;i<pose_entries.rows;i++){
                std::vector<cv::Point> pose_keypoints(Pose::num_kpts,cv::Point(-1,-1));
                int valid_num = 0;
                for(int kpt_id = 0; kpt_id < Pose::num_kpts; kpt_id++){
                    if(pose_entries.at<float>(i, kpt_id) != -1){
                        pose_keypoints[kpt_id]=all_keypoints[(int)pose_entries.at<float>(i,kpt_id)].point;
                        valid_num++;
                    }
                }
                if(valid_num >= Pose::num_kpts/2){
                    Pose p(pose_keypoints,pose_entries.at<float>(i,18));
                    current_poses.push_back(p);
                }
            }
            return current_poses;
        }

        cv::Mat pad_resize(const cv::Mat& input_image, int new_height, int new_width){
            cv::Mat ret= input_image.channels()>=3 ? cv::Mat(cv::Size(new_width, new_height), CV_8UC3, cv::Scalar(128,128,128)) : 
                cv::Mat(cv::Size(new_width, new_height), CV_8UC1, cv::Scalar(0));
            float old_height = input_image.rows, old_width = input_image.cols;
            float ratio = std::min(new_height/old_height, new_width/old_width);
            cv::Size resize_size = cv::Size(old_width * ratio, old_height * ratio);
            cv::Mat temp_img;
            int top = (new_height - resize_size.height)/2;
            int left = (new_width - resize_size.width)/2;
            
            cv::resize(input_image, temp_img, resize_size, 0, 0, cv::INTER_LINEAR);

            cv::Mat roi = ret(cv::Rect(left, top, resize_size.width, resize_size.height));
            temp_img.copyTo(roi);
            this->resize_ratio = ratio;
            this->resize_left_padding = left;
            this->resize_top_padding = top;
            return ret;
        }

        template<typename T> std::vector<int> argsort(const std::vector<T>& array){
    	    const int array_len(array.size());
    	    std::vector<int> array_index(array_len, 0);
	        for (int i = 0; i < array_len; ++i)
		        array_index[i] = i;

        	std::sort(array_index.begin(), array_index.end(),
	        	[&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);});
    	    return array_index;
        }

        void group_keypoints(const std::vector<std::vector<keypoint>>& all_keypoints_by_type, const std::vector<cv::Mat> &pafs_channels, 
                        cv::Mat &filtered_entries,std::vector<keypoint> &all_keypoints,
                        int pose_entry_size=20, float min_paf_score=0.05){
            all_keypoints.clear();
            for(int i=0;i<all_keypoints_by_type.size();i++)
                for(int j=0;j<all_keypoints_by_type[i].size();j++)
                    all_keypoints.push_back(all_keypoints_by_type[i][j]);
            
            int points_per_limb = 10;
            
            cv::Mat pose_entries;
            for(int part_id = 0 ;part_id<BODY_PARTS_PAF_IDS.size();part_id++)
            {
                std::vector<cv::Mat> part_pafs{pafs_channels[BODY_PARTS_PAF_IDS[part_id][0]], pafs_channels[BODY_PARTS_PAF_IDS[part_id][1]]};
                std::vector<keypoint> kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]];
                std::vector<keypoint> kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]];

                int n = kpts_a.size();
                int m = kpts_b.size();
                if(n==0 || m==0)
                    continue;
                
                cv::Mat a(m, n, CV_32FC2);
                for(int i=0;i<m;i++)
                    for(int j=0;j<n;j++)
                        a.at<cv::Vec2f>(i,j)=cv::Vec2f(kpts_a[j].point.x,kpts_a[j].point.y);

                cv::Mat b(m, n, CV_32FC2);
                for(int i=0;i<m;i++)
                    for(int j=0;j<n;j++)
                        b.at<cv::Vec2f>(i,j)=cv::Vec2f(kpts_b[i].point.x,kpts_b[i].point.y);

                cv::Mat vec_raw = cv::Mat(b - a).reshape(0, m * n);

                cv::Mat steps = 1.0 / (points_per_limb -1) * vec_raw;
                cv::Mat steps_broadcast(steps.rows,points_per_limb,steps.type());
                for(int j=0;j<points_per_limb;j++)
                    steps.copyTo(steps_broadcast.col(j));
                cv::Mat grid(steps_broadcast.rows,points_per_limb,steps_broadcast.type());
                for(int i=0;i<grid.rows;i++)
                    for(int j=0;j<grid.cols;j++)
                        grid.at<cv::Vec2f>(i,j)=cv::Vec2f(j,j);

                a = a.reshape(0, m*n);
                cv::Mat a_broadcast(a.rows,points_per_limb,a.type());
                for(int j=0;j<a_broadcast.cols;j++)
                    a.copyTo(a_broadcast.col(j));
                cv::Mat points = steps_broadcast.mul(grid) + a_broadcast;
                points.convertTo(points,CV_32S);

                std::vector<cv::Mat> xy_channels;
                cv::split(points,xy_channels);
                cv::Mat x = xy_channels[0].reshape(0, xy_channels[0].total());
                cv::Mat y = xy_channels[1].reshape(0, xy_channels[1].total());

                cv::Mat field(x.rows,1,CV_32FC2);
                for(int i=0;i<x.rows;i++)
                {
                    int x_index = x.at<int>(i,0);
                    int y_index = y.at<int>(i,0);
                    field.at<cv::Vec2f>(i, 0) = 
                        cv::Vec2f(part_pafs[0].at<float>(y_index,x_index), part_pafs[1].at<float>(y_index,x_index));
                }
                field = field.reshape(0, field.rows*field.cols/points_per_limb);

                cv::Mat vec_norm = vec_raw.clone();
                for(int i=0;i<vec_norm.rows;i++)
                    for(int j=0;j<vec_norm.cols;j++)
                    {
                        float norm = sqrt(pow(vec_norm.at<cv::Vec2f>(i,j)[0],2)+pow(vec_norm.at<cv::Vec2f>(i,j)[1],2));
                        vec_norm.at<cv::Vec2f>(i,j)=cv::Vec2f(1/(norm+1e-6),1/(norm+1e-6));
                    }
                cv::Mat vec = vec_raw.mul(vec_norm);
                cv::Mat vec_broadcast(vec.rows,points_per_limb,vec.type());
                for(int i=0;i<vec_broadcast.cols;i++)
                    vec.copyTo(vec_broadcast.col(i));
                
                cv::Mat affinity_scores(field.rows,field.cols,CV_32FC1);
                cv::Mat field_dot_vec = field.mul(vec_broadcast);
                for(int i=0;i<affinity_scores.rows;i++)
                    for(int j=0;j<affinity_scores.cols;j++)
                        affinity_scores.at<float>(i,j) = field_dot_vec.at<cv::Vec2f>(i,j)[0] + field_dot_vec.at<cv::Vec2f>(i,j)[1];
                
                affinity_scores = affinity_scores.reshape(0 ,affinity_scores.total()/points_per_limb);
                cv::Mat valid_affinity_scores = (affinity_scores > min_paf_score);
                valid_affinity_scores.convertTo(valid_affinity_scores, CV_32F, 1.0 / 255);
                cv::Mat valid_num;
    			cv::reduce(valid_affinity_scores, valid_num, 1, cv::REDUCE_SUM, CV_32F);
                cv::reduce(affinity_scores.mul(valid_affinity_scores),affinity_scores,1,cv::REDUCE_SUM,CV_32F);                
                affinity_scores = affinity_scores.mul(1 / (valid_num + 1e-6));
                cv::Mat success_ratio = valid_num / points_per_limb;

                cv::Mat valid_limbs;
                cv::findNonZero((affinity_scores>0) & (success_ratio>=0.8), valid_limbs);
                std::vector<cv::Mat> valid_limbs_channels;

                cv::split(valid_limbs,valid_limbs_channels);
                if(valid_limbs_channels.size()==0)
                    continue;
                valid_limbs = valid_limbs_channels[1];


                std::vector<int> b_idx(valid_limbs.rows),a_idx(valid_limbs.rows);

                for(int i=0;i<valid_limbs.rows;i++){
                    b_idx[i]=valid_limbs.at<int>(i)/n;
                    a_idx[i]=valid_limbs.at<int>(i)%n;
                }

                std::vector<float> affinity_scores_valid_limbs_vector;
                for(int i=0;i<valid_limbs.rows;i++){
                    affinity_scores_valid_limbs_vector.push_back(affinity_scores.at<float>(valid_limbs.at<int>(i)));
                }
                connections_nms(a_idx,b_idx,affinity_scores_valid_limbs_vector);

                int connections_len = affinity_scores_valid_limbs_vector.size();
                if(connections_len==0)
                    continue;

                if(part_id==0){
                    pose_entries = cv::Mat(connections_len,pose_entry_size,CV_32FC1,1);
                    pose_entries *=-1;
                    for(int i=0;i<connections_len;i++){
                        pose_entries.at<float>(i, BODY_PARTS_KPT_IDS[0][0]) = kpts_a[a_idx[i]].idx;
                        pose_entries.at<float>(i, BODY_PARTS_KPT_IDS[0][1]) = kpts_b[b_idx[i]].idx;
                        pose_entries.at<float>(i, pose_entries.cols - 1) = 2;
                        pose_entries.at<float>(i, pose_entries.cols - 2) = all_keypoints[kpts_a[a_idx[i]].idx].conf + all_keypoints[kpts_b[b_idx[i]].idx].conf + affinity_scores_valid_limbs_vector[i];
                    }
                }
                else if (part_id==17 || part_id==18){
                    int kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0];
                    int kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1];
                    for(int i=0;i<connections_len;i++)
                        for(int j=0;j<pose_entries.rows;j++)
                            if(pose_entries.at<float>(j,kpt_a_id) == kpts_a[a_idx[i]].idx && pose_entries.at<float>(j,kpt_b_id) == -1)
                                pose_entries.at<float>(j,kpt_b_id) =  kpts_b[b_idx[i]].idx;
                            else if(pose_entries.at<float>(j,kpt_b_id) == kpts_b[b_idx[i]].idx && pose_entries.at<float>(j,kpt_a_id) == -1)
                                pose_entries.at<float>(j,kpt_a_id) == kpts_a[a_idx[i]].idx;
                    continue;
                }
                else{
                    int kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0];
                    int kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1];
                    for(int i=0;i<connections_len;i++){
                        int num = 0;
                        for(int j=0;j<pose_entries.rows;j++){
                            if(pose_entries.at<float>(j,kpt_a_id) == kpts_a[a_idx[i]].idx){
                                pose_entries.at<float>(j,kpt_b_id) = kpts_b[b_idx[i]].idx;
                                num++;
                                pose_entries.at<float>(j, pose_entries.cols - 1) +=1;
                                pose_entries.at<float>(j, pose_entries.cols - 2) += all_keypoints[kpts_b[b_idx[i]].idx].conf + affinity_scores_valid_limbs_vector[i];
                            }
                        }
                        if(num==0){
                            cv::Mat pose_entry(1, pose_entry_size,CV_32FC1, 1);
                            pose_entry *= -1;
                            pose_entry.at<float>(0, kpt_a_id) = kpts_a[a_idx[i]].idx;
                            pose_entry.at<float>(0, kpt_b_id) = kpts_b[b_idx[i]].idx;
                            pose_entry.at<float>(0, pose_entry.cols - 1) = 2;
                            pose_entry.at<float>(0, pose_entry.cols - 2) = all_keypoints[kpts_a[a_idx[i]].idx].conf + all_keypoints[kpts_b[b_idx[i]].idx].conf + affinity_scores_valid_limbs_vector[i];
                            pose_entries.push_back(pose_entry);
                        }
                    }
                }
            }

            filtered_entries = cv::Mat(0,pose_entry_size,CV_32FC1);
            for(int i=0;i<pose_entries.rows;i++){
                if(pose_entries.at<float>(i,pose_entries.cols-1)<3 || pose_entries.at<float>(i,pose_entries.cols-2)/pose_entries.at<float>(i,pose_entries.cols-1)<0.2)
                    continue;
                filtered_entries.push_back(pose_entries.row(i));
            }
        }

        int extract_keypoints(cv::Mat& heatmap, std::vector<std::vector<keypoint>>& all_keypoints, int& total_keypoint_num)
        {

            cv::threshold(heatmap,heatmap,0.1,0,cv::THRESH_TOZERO);
            cv::Mat heatmap_with_borders;
            cv::copyMakeBorder(heatmap, heatmap_with_borders, 2, 2, 2, 2,cv::BORDER_CONSTANT);
            
            cv::Mat heatmap_center = heatmap_with_borders(cv::Range(1, heatmap_with_borders.rows - 1), cv::Range(1, heatmap_with_borders.cols - 1));
            cv::Mat heatmap_left = heatmap_with_borders(cv::Range(1, heatmap_with_borders.rows - 1), cv::Range(2, heatmap_with_borders.cols));
            cv::Mat heatmap_right = heatmap_with_borders(cv::Range(1, heatmap_with_borders.rows - 1), cv::Range(0, heatmap_with_borders.cols - 2));
            cv::Mat heatmap_up = heatmap_with_borders(cv::Range(2, heatmap_with_borders.rows), cv::Range(1, heatmap_with_borders.cols - 1));
            cv::Mat heatmap_down = heatmap_with_borders(cv::Range(0, heatmap_with_borders.rows - 2), cv::Range(1, heatmap_with_borders.cols - 1));

            cv::Mat heatmap_peaks = (heatmap_center > heatmap_left)
                    & (heatmap_center > heatmap_right)
                    & (heatmap_center > heatmap_up)
                    & (heatmap_center > heatmap_down);
            heatmap_peaks = heatmap_peaks(cv::Range(1, heatmap_center.rows - 1), cv::Range(1, heatmap_center.cols - 1));
            cv::Mat keypoints_mat;
            cv::findNonZero(heatmap_peaks,keypoints_mat);
            std::vector<cv::Point> keypoints(keypoints_mat.total());
            for(int i=0;i<keypoints_mat.total();i++)
                keypoints[i] = keypoints_mat.at<cv::Point>(i);
            sort(keypoints.begin(),keypoints.end(),
                [](const cv::Point& a,const cv::Point& b)
                {return a.x<b.x;});
            std::vector<int> suppressed(keypoints.size());
            std::vector<keypoint> keypoints_with_score_and_id;
            int keypoint_num = 0;
            for(int i=0;i<keypoints.size();i++){
                if(suppressed[i])
                    continue;
                for(int j=i+1;j<keypoints.size();j++){
                    if(sqrt(pow(keypoints[i].x - keypoints[j].x,2)+pow(keypoints[i].y - keypoints[j].y,2))<6)
                        suppressed[j] = 1;
                }
                keypoints_with_score_and_id.push_back({keypoints[i],heatmap.at<float>(keypoints[i].y,keypoints[i].x),total_keypoint_num + keypoint_num});
                keypoint_num++;
            }
            all_keypoints.push_back(keypoints_with_score_and_id);
            
            return keypoint_num;


        }
        
        void connections_nms(std::vector<int>& a_idx,std::vector<int>& b_idx,std::vector<float>& affinity_scores)
        {
            std::vector<int> order = argsort(affinity_scores);
            reverse(order.begin(),order.end());

            std::vector<int> new_a_idx,new_b_idx;
            std::vector<float> new_affinity_scores;
            for(int i=0;i<order.size();i++)
            {
                new_a_idx.push_back(a_idx[order[i]]);
                new_b_idx.push_back(b_idx[order[i]]);
                new_affinity_scores.push_back(affinity_scores[order[i]]);
            }

            std::vector<int> idx;
            std::set<int> has_kpt_a;
            std::set<int> has_kpt_b;

            for(int i=0;i<order.size();i++){
                if(has_kpt_a.find(a_idx[i])==has_kpt_a.end() && has_kpt_b.find(b_idx[i])==has_kpt_b.end()){
                    idx.push_back(i);
                    has_kpt_a.insert(a_idx[i]);
                    has_kpt_b.insert(b_idx[i]);
                }
            }
            a_idx.resize(idx.size());
            b_idx.resize(idx.size());
            affinity_scores.resize(idx.size());
            for(int i=0;i<idx.size();i++)
            {
                a_idx[i]=new_a_idx[idx[i]];
                b_idx[i]=new_b_idx[idx[i]];
                affinity_scores[i]=new_affinity_scores[idx[i]];
            }
        }

    };
}
