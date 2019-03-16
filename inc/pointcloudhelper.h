#ifndef POINTCLOUDHELPER_H
#define POINTCLOUDHELPER_H

#include "global.h"
#include "object.h"
#include "visualizer.h"

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

static void ccl(cv::Mat &map, std::vector<int> &label)
{
    cv::Mat m;
    cv::cvtColor(map, m, CV_BGR2GRAY);
	// printf("[pointcloud_labelling] map cols %d, rows %d\n", m.cols, m.cols);

    std::vector<bool> visited(m.rows * m.cols, 0);
    std::vector<int> area(m.rows * m.cols, 0);

	int label_cnt = 0;

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
		std::vector<int> label(lidarmap.rows * lidarmap.cols, 0);
		ccl(lidarmap, label);
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

static void detection_core(std::vector<cv::Point> &roi, 
						   std::vector<int> &label, 
						   std::vector<BoxObject> &lidarobjs,
						   int step)
{
	// analyse which label does roi contain mostly
	std::map<int, int> label_cnt;
	for (auto pt : roi)
	{
		int u = pt.x;
		int v = pt.y;
		int label_tmp = label[v*step + u];

		if (label_cnt.find(label_tmp) == label_cnt.end())
			label_cnt[label_tmp] = 1;
		else
			label_cnt[label_tmp]++;
	}
	int max_cnt = 0;
	int max_cnt_label = 0;
	for (auto each_label_cnt : label_cnt)
	{
		if (each_label_cnt.second > max_cnt)
		{
			max_cnt = each_label_cnt.second;
			max_cnt_label = each_label_cnt.first;
		}
	}

	// rearrange roi pointclouds with one main label
	std::vector<cv::Point> roi_old;
	roi_old.swap(roi);
	for (auto pt : roi_old)
	{
		int u = pt.x;
		int v = pt.y;
		int label_tmp = label[v*step + u];

		if (label_tmp == max_cnt_label)
			roi.push_back(pt);
	}

	// draw box
	// naive box to include roi pointcloud, not in minimun size
	// TODO minimum size box
	float min_x = FLT_MAX, min_y = FLT_MAX;
	float max_x = -FLT_MAX, max_y = -FLT_MAX;

	for (auto pt : roi)
	{
		int u = pt.x;
		int v = pt.y;

		float rx = map_range_length - v * map_scale;
		float ry = u * map_scale - map_range_width;

		// printf("[detection_core] one point rx %f, ry %f, u %d, v %d\n",
		// 	   rx, ry, u, v);

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

	// summary
	BoxObject lidarobj;
	lidarobj.rx = (min_x + max_x) / 2;
	lidarobj.ry = (min_y + max_y) / 2;
	lidarobj.vx = lidarobj.vy = 0;

	float size_l = max_x - min_x;
	size_l = std::max(box_object_len, size_l);
	float size_w = max_y - min_y;
	size_w = std::max(box_object_wid, size_w);
	lidarobj.corner[0] = cv::Point2f(size_l/2, -size_w/2);
    lidarobj.corner[1] = cv::Point2f(-size_l/2, -size_w/2);
    lidarobj.corner[2] = cv::Point2f(-size_l/2, size_w/2);
    lidarobj.corner[3] = cv::Point2f(size_l/2, size_w/2);
	
	lidarobj.yaw = Eigen::Quaternionf(1, 0, 0, 0);
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
		std::vector<int> label(lidarmap.rows * lidarmap.cols, 0);
		ccl(lidarmap, label);

		for (auto radarobj : filtered_radarobjs)
		{
			float rx = radarobj.rx;
			float ry = radarobj.ry;
			
			// find 20*20 roi
			std::vector<cv::Point> roi;
			for (float i=-20/2; i<=20/2; ++i)
			{
				for (float j=-20/2; j<=20/2; ++j)
				{
					float rx_ = rx + i;
					float ry_ = ry + j;

					cv::Point pt(ry_ / map_scale + map_range_width / map_scale,
						map_range_length / map_scale - rx_ / map_scale);

					if (label[pt.y*lidarmap.cols + pt.x])
						roi.push_back(pt);
				}
			}

			// generate lidar box from roi pointclouds
			if (roi.size())
				detection_core(roi, label, lidarobjs, lidarmap.cols);
		}
	}
}

#endif // POINTCLOUDHELPER_H
