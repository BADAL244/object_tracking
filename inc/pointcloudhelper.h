#ifndef POINTCLOUDHELPER_H
#define POINTCLOUDHELPER_H

#include "global.h"
#include "object.h"
#include "visualizer.h"

#include "dbscan.h"
#include "meanshift.h"

const float ccl_win_w = 2 / map_scale;
const float ccl_win_h = 2 / map_scale;

static cv::Scalar label2color(int label, int label_num)
{
	cv::Scalar color(0, 0, 0);
    if (!label_num)  return color;

	label *= (256 / (label_num));
	if (!label)
	{
		color[0] = 0;
		color[1] = 0;
		color[2] = 0;
	}
	else if (label <= 51)
	{
		color[0] = 255;
		color[1] = label * 5;
		color[2] = 0;
	}
	else if (label <= 102)
	{
		label -= 51;
		color[0] = 255 - label * 5;
		color[1] = 255;
		color[2] = 0;
	}
	else if (label <= 153)
	{
		label -= 102;
		color[0] = 0;
		color[1] = 255;
		color[2] = label * 5;
	}
	else if (label <= 204)
	{
		label -= 153;
		color[0] = 0;
		color[1] = 255 - uchar(128.0*label / 51.0 + 0.5);
		color[2] = 255;
	}
	else
	{
		label -= 204;
		color[0] = 0;
		color[1] = 127 - uchar(127.0*label / 51.0 + 0.5);
		color[2] = 255;
	}

	return color;
}

static void ccl_dfs(int row, int col, cv::Mat &m, 
                    std::vector<bool> &visited, 
                    std::vector<int> &label, 
                    std::vector<int> &area,  
                    int label_cnt)
{
	visited[row * m.cols + col] = 1;

    for (int i=-ccl_win_w/2; i<=ccl_win_w/2; ++i)
    {
        for (int j=-ccl_win_h/2; j<=ccl_win_h/2; ++j)
        {
            int u = std::max(0, col+i);
            u = std::min(m.cols-1, u);
            int v = std::max(0, row+j);
            v = std::min(m.rows-1, v);

            if (!visited[v*m.cols + u] && (m.at<uchar>(row, col) == m.at<uchar>(v, u)))
            {
                label[v*m.cols + u] = label_cnt;
                ++area[label_cnt];
                ccl_dfs(v, u, m, visited, label, area, label_cnt);
            }
        }
    }
}

static void ccl(cv::Mat &map, std::vector<int> &label, int &label_cnt)
{
    cv::Mat m;
    cv::cvtColor(map, m, CV_BGR2GRAY);
	// printf("[pointcloud_labelling] map cols %d, rows %d\n", m.cols, m.cols);

    std::vector<bool> visited(m.rows * m.cols, 0);
    std::vector<int> area(m.rows * m.cols, 0);

	// int label_cnt = 0;
	label_cnt = 0;

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			if (!m.at<uchar>(i, j))  // no lidar -> label 0
			{
				label[i*m.cols + j] = 0;
				area[0]++;
				continue;
			}
			if (visited[i*m.cols + j])
			{
				continue;
			}

			label[i*m.cols + j] = ++label_cnt;
			++area[label_cnt];
			ccl_dfs(i, j, m, visited, label, area, label_cnt);
		}
	}

	printf("[pointcloud_labelling] there is %d cluster at all\n", label_cnt);
	// for (int i = 0; i <= label_cnt; i++)
	// {
	// 	printf("[pointcloud_labelling] cluster %d: %d\n", i, area[i]);
	// }

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			cv::Scalar color = label2color(label[i*m.cols + j], label_cnt);
			map.at<cv::Vec3b>(i,j)[0] = color[0];
			map.at<cv::Vec3b>(i,j)[1] = color[1];
			map.at<cv::Vec3b>(i,j)[2] = color[2];
		}
	}
}

void pointcloud_labelling(std::vector<LidarPoint> &lidarpts)
{
    Visualizer pcviser;
    pcviser.Init();
    pcviser.DrawLidarPts(lidarpts, cv::Scalar(0,255,255));
    cv::Mat lidarmap = pcviser.GetMap();

	printf("[pointcloud_labelling] %zu pointclouds received\n", lidarpts.size());
	if (lidarpts.size())
	{
		int label_cnt = 0;
		std::vector<int> label(lidarmap.rows * lidarmap.cols, 0);
		ccl(lidarmap, label, label_cnt);

		// draw cluster center in white
		if (label_cnt)
		{
			std::vector<LidarPoint> cluster_center(label_cnt+1, {FLT_MAX, FLT_MAX});
			for (auto lpt : lidarpts)
			{
				cv::Point map_pt;
				map_pt.x = lpt.ry / map_scale + map_range_width / map_scale,
				map_pt.y = map_range_length / map_scale - lpt.rx / map_scale;

				map_pt.x = std::max(map_pt.x, 0);
				map_pt.x = std::min(map_pt.x, lidarmap.cols-1);
				map_pt.y = std::max(map_pt.y, 0);
				map_pt.y = std::min(map_pt.y, lidarmap.rows-1);

				assert(label[map_pt.y*lidarmap.cols+map_pt.x]);  // need carefully check

				LidarPoint nearest_lpt = cluster_center[label[map_pt.y*lidarmap.cols+map_pt.x]];
				float nearest_dist = sqrt(pow(nearest_lpt.rx, 2)+pow(nearest_lpt.ry, 2));
				float tmp_dist = sqrt(pow(lpt.rx, 2)+pow(lpt.ry, 2));
				if (tmp_dist < nearest_dist)
					cluster_center[label[map_pt.y*lidarmap.cols+map_pt.x]] = lpt;
			}

			for (auto lpt : cluster_center)
			{
				cv::Point map_pt;
				map_pt.x = lpt.ry / map_scale + map_range_width / map_scale,
				map_pt.y = map_range_length / map_scale - lpt.rx / map_scale;

				map_pt.x = std::max(map_pt.x, 0);
				map_pt.x = std::min(map_pt.x, lidarmap.cols-1);
				map_pt.y = std::max(map_pt.y, 0);
				map_pt.y = std::min(map_pt.y, lidarmap.rows-1);

				cv::Scalar color(255,255,255);
				cv::circle(lidarmap, map_pt, 2, color, 2);
			}
		}
	}
	
	// // test ccl
	// cv::Mat lidarmap(500, 500, CV_8UC3, cv::Scalar(0,0,0));
	// cv::rectangle(lidarmap, cv::Rect(100, 100, 100, 100), cv::Scalar(255, 255, 255));
	// ccl(lidarmap);
	// printf("ccl end\n");

    cv::namedWindow("pointcloud_label");
	cv::imshow("pointcloud_label", lidarmap);
    cv::waitKey(5);
}

void pointcloud_labelling_try_dbscan(std::vector<LidarPoint> &lidarpts)
{
    Visualizer pcviser;
    pcviser.Init();
    pcviser.DrawLidarPts(lidarpts, cv::Scalar(0,255,255));
    cv::Mat lidarmap = pcviser.GetMap();

	printf("[pointcloud_labelling_try_dbscan] %zu pointclouds received\n", lidarpts.size());
	if (lidarpts.size())
	{
		std::vector<std::vector<int> > dbscan_res = dbscan_entry(lidarpts.size(), 1.0, 1, lidarpts);
		
		printf("[pointcloud_labelling_try_dbscan] there is %zu cluster at all\n", dbscan_res.size());

		lidarmap = cv::Mat::zeros(lidarmap.rows, lidarmap.cols, CV_8UC3);
		for (int i=0; i<dbscan_res.size(); ++i)
		{
			for (int j=0; j<dbscan_res[i].size(); ++j)
			{
				LidarPoint lpt = lidarpts[dbscan_res[i][j]];
				cv::Point map_pt;
				map_pt.x = lpt.ry / map_scale + map_range_width / map_scale,
            	map_pt.y = map_range_length / map_scale - lpt.rx / map_scale;
				
				map_pt.x = std::max(map_pt.x, 0);
            	map_pt.x = std::min(map_pt.x, lidarmap.cols-1);
            	map_pt.y = std::max(map_pt.y, 0);
            	map_pt.y = std::min(map_pt.y, lidarmap.rows-1);

				cv::Scalar color = label2color(i+1, dbscan_res.size());
				lidarmap.at<cv::Vec3b>(map_pt.y, map_pt.x)[0] = color[0];
				lidarmap.at<cv::Vec3b>(map_pt.y, map_pt.x)[1] = color[1];
				lidarmap.at<cv::Vec3b>(map_pt.y, map_pt.x)[2] = color[2];
			}
		}
		printf("[pointcloud_labelling_try_dbscan] cluster display ok\n");
	}
	
    cv::namedWindow("pointcloud_label");
	cv::imshow("pointcloud_label", lidarmap);
    cv::waitKey(5);
}

void pointcloud_labelling_try_meanshift(std::vector<LidarPoint> &lidarpts)
{
    Visualizer pcviser;
    pcviser.Init();
    pcviser.DrawLidarPts(lidarpts, cv::Scalar(0,255,255));
    cv::Mat lidarmap = pcviser.GetMap();

	printf("[pointcloud_labelling_try_meanshift] %zu pointclouds received\n", lidarpts.size());
	if (lidarpts.size())
	{
		std::vector<Cluster> meanshift_res = meanshift_entry(lidarpts, 1);
		
		printf("[pointcloud_labelling_try_meanshift] there is %zu cluster at all\n", meanshift_res.size());

		lidarmap = cv::Mat::zeros(lidarmap.rows, lidarmap.cols, CV_8UC3);
		for (int i=0; i<meanshift_res.size(); ++i)
		{
			for (int j=0; j<meanshift_res[i].original_points.size(); ++j)
			{
				std::vector<double> lpt = meanshift_res[i].original_points[j];

				cv::Point map_pt;
				map_pt.x = lpt[1] / map_scale + map_range_width / map_scale,
            	map_pt.y = map_range_length / map_scale - lpt[0] / map_scale;
				
				map_pt.x = std::max(map_pt.x, 0);
            	map_pt.x = std::min(map_pt.x, lidarmap.cols-1);
            	map_pt.y = std::max(map_pt.y, 0);
            	map_pt.y = std::min(map_pt.y, lidarmap.rows-1);

				cv::Scalar color = label2color(i+1, meanshift_res.size());
				lidarmap.at<cv::Vec3b>(map_pt.y, map_pt.x)[0] = color[0];
				lidarmap.at<cv::Vec3b>(map_pt.y, map_pt.x)[1] = color[1];
				lidarmap.at<cv::Vec3b>(map_pt.y, map_pt.x)[2] = color[2];
			}
		}
		printf("[pointcloud_labelling_try_meanshift] cluster display ok\n");
	}
	
    cv::namedWindow("pointcloud_label");
	cv::imshow("pointcloud_label", lidarmap);
    cv::waitKey(5);
}

static void detection_core(std::vector<LidarPoint> &roi,
						   std::vector<int> &label, 
						   std::vector<BoxObject> &lidarobjs,
						   int step)
{
	// if roi contains one kind of label
	std::map<int, int> label_cnt;
	for (auto lpt : roi)
	{
		int u = lpt.ry / map_scale + map_range_width / map_scale;
		int v = map_range_length / map_scale - lpt.rx / map_scale;
		int label_tmp = label[v*step + u];

		if (label_cnt.find(label_tmp) == label_cnt.end())
			label_cnt[label_tmp] = 1;
		else
			label_cnt[label_tmp]++;
	}
	if (label_cnt.size() != 1)  return;

	// draw box
	// naive box to include roi pointclouds as a reference, not in minimum size
	float min_x = FLT_MAX, min_y = FLT_MAX;
	float max_x = -FLT_MAX, max_y = -FLT_MAX;

	for (auto lpt : roi)
	{
		float rx = lpt.rx;
		float ry = lpt.ry;

		if (rx > max_x)
			max_x = rx;
		if (rx < min_x)
			min_x = rx;
		if (ry > max_y)
			max_y = ry;
		if (ry < min_y)
			min_y = ry;
	}

	// printf("[detection_core] min_x %f, max_x %f, min_y %f, max_y %f\n",
	// 	   min_x, max_x, min_y, max_y);

	float ctr_rx = (min_x + max_x) / 2;
	float ctr_ry = (min_y + max_y) / 2;
	float min_area = fabs((max_x - min_x) * (max_y - min_y));
	float size_l = fabs(max_x - min_x);
	size_l = std::max(box_object_len, size_l);
	float size_w = fabs(max_y - min_y);
	size_w = std::max(box_object_wid, size_w);

	float final_yaw = 0;
	float final_size_l = size_l;
	float final_size_w = size_w;

	// find box with minimum size
	Eigen::MatrixXd point_matrix = Eigen::MatrixXd::Zero(2, roi.size());
	for (int i=0; i<roi.size(); ++i)
	{
		LidarPoint lpt = roi[i];
		float rx = lpt.rx;
		float ry = lpt.ry;

		point_matrix(0, i) = rx - ctr_rx;
		point_matrix(1, i) = ry - ctr_ry;
	}

	float min_score = FLT_MAX;
	for (float yaw = 0; yaw < 180; yaw += 5)
	{
		// avoid ambiguous and repeatness
		if ((180 - yaw > 45) && (180 - yaw < 135))  continue;

		float yaw_rad = deg2rad(yaw);
		Eigen::Matrix2d R;
		R << cos(yaw_rad), -sin(yaw_rad),
		     sin(yaw_rad), cos(yaw_rad);
		
		Eigen::MatrixXd rotated_point_matrix = R * point_matrix;

		min_x = FLT_MAX, min_y = FLT_MAX;
		max_x = -FLT_MAX, max_y = -FLT_MAX;
		for (int i=0; i<roi.size(); ++i)
		{
			float rx = rotated_point_matrix(0, i) + ctr_rx;
			float ry = rotated_point_matrix(1, i) + ctr_ry;

			if (rx > max_x)
				max_x = rx;
			if (rx < min_x)
				min_x = rx;
			if (ry > max_y)
				max_y = ry;
			if (ry < min_y)
				min_y = ry;
		}
		float area = fabs((max_x - min_x) * (max_y - min_y));

		if (true)
		{
			// compute fitted scores using size and pointclouds in roi can get better yaw
			float score = 0;
			for (int i=0; i<roi.size(); ++i)
			{
				float rx = rotated_point_matrix(0, i) + ctr_rx;
				float ry = rotated_point_matrix(1, i) + ctr_ry;

				float score_rx = std::min(fabs(rx - max_x), fabs(rx - min_x));
				float score_ry = std::min(fabs(ry - max_y), fabs(ry - min_y));
				score += pow(std::min(score_rx, score_ry), 2);
			}
			if (score < min_score)
			{
				min_score = score;

				final_yaw = 180 - yaw;
				final_size_l = fabs(max_x - min_x);
				// final_size_l = std::max(box_object_len, final_size_l);
				final_size_w = fabs(max_y - min_y);
				// final_size_w = std::max(box_object_wid, final_size_w);
			}
		}
		else
		{
			if (area < min_area)
			{
				min_area = area;

				final_yaw = 180 - yaw;
				final_size_l = fabs(max_x - min_x);
				// final_size_l = std::max(box_object_len, final_size_l);
				final_size_w = fabs(max_y - min_y);
				// final_size_w = std::max(box_object_wid, final_size_w);

				if (final_size_l < box_object_len
				    && final_size_l < box_object_wid
					&& final_size_w < box_object_len
					&& final_size_w < box_object_wid)
					return;
			}
		}
	}

	final_yaw = deg2rad(final_yaw);

	// summary
	BoxObject lidarobj;
	lidarobj.rx = ctr_rx;
	lidarobj.ry = ctr_ry;
	lidarobj.vx = lidarobj.vy = 0;
	lidarobj.corner[0] = cv::Point2f(final_size_l/2, -final_size_w/2);
    lidarobj.corner[1] = cv::Point2f(-final_size_l/2, -final_size_w/2);
    lidarobj.corner[2] = cv::Point2f(-final_size_l/2, final_size_w/2);
    lidarobj.corner[3] = cv::Point2f(final_size_l/2, final_size_w/2);
	lidarobj.yaw = Eigen::Quaternionf(cos(final_yaw/2), 0, 0, sin(final_yaw/2));
	lidarobjs.push_back(lidarobj);
}

void lidar_object_detection(std::vector<LidarPoint> &lidarpts, 
							std::vector<BoxObject> &filtered_radarobjs,
							std::vector<BoxObject> &lidarobjs)
{
	lidarobjs.clear();
	printf("[pointcloud_labelling] %zu pointclouds received\n", lidarpts.size());
	if (lidarpts.size())
	{
		Visualizer pcviser;
    	pcviser.Init();
    	pcviser.DrawLidarPts(lidarpts, cv::Scalar(0,255,255));
		
    	cv::Mat lidarmap = pcviser.GetMap();
		int label_cnt = 0;
		std::vector<int> label(lidarmap.rows * lidarmap.cols, 0);
		ccl(lidarmap, label, label_cnt);

		for (auto radarobj : filtered_radarobjs)
		{
			float rx = radarobj.rx;
			float ry = radarobj.ry;
			
			// find 15m*15m roi
			std::vector<LidarPoint> roi;
			for (float i=-15.0/2; i<=15.0/2; i+=0.1)
			{
				for (float j=-15.0/2; j<=15.0/2; j+=0.1)
				{
					float rx_ = rx + i;
					float ry_ = ry + j;

					cv::Point pt(ry_ / map_scale + map_range_width / map_scale,
						map_range_length / map_scale - rx_ / map_scale);

					if (label[pt.y*lidarmap.cols + pt.x])
					{
						LidarPoint lpt{rx_, ry_};
						roi.push_back(lpt);
					}
				}
			}

			// generate lidar box from roi pointclouds
			if (roi.size())
				detection_core(roi, label, lidarobjs, lidarmap.cols);
		}
	}
}

#endif // POINTCLOUDHELPER_H
