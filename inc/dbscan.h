// https://github.com/bowbowbow/DBSCAN

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <map>

#include "global.h"

const int NOISE = -2;
const int NOT_CLASSIFIED = -1;

class Point {
public:
    double x, y;
    int ptsCnt, cluster;
    double getDis(const Point & ot) {
        return sqrt((x-ot.x)*(x-ot.x)+(y-ot.y)*(y-ot.y));
    }
};

class DBSCAN {
public:
    int n, minPts;
    double eps;
    std::vector<Point> points;
    int size;
    std::vector<std::vector<int> > adjPoints;
    std::vector<bool> visited;
    std::vector<std::vector<int> > cluster;
    int clusterIdx;
    
    DBSCAN(int n, double eps, int minPts, std::vector<Point> points) {
        this->n = n;
        this->eps = eps;
        this->minPts = minPts;
        this->points = points;
        this->size = (int)points.size();
        adjPoints.resize(size);
        this->clusterIdx=-1;
    }
    void run () {
        checkNearPoints();
        printf("checkNearPoints ok\n");
        
        for(int i=0;i<size;i++) {
            if(points[i].cluster != NOT_CLASSIFIED) continue;
            
            if(isCoreObject(i)) {
                dfs(i, ++clusterIdx);
            } else {
                points[i].cluster = NOISE;
            }
        }
        printf("dfs ok\n");
        
        cluster.resize(clusterIdx+1);
        for(int i=0;i<size;i++) {
            if(points[i].cluster != NOISE) {
                cluster[points[i].cluster].push_back(i);
            }
        }
        printf("result ok\n");
    }
    
    void dfs (int now, int c) {
        points[now].cluster = c;
        if(!isCoreObject(now)) return;
        
        for(auto&next:adjPoints[now]) {
            if(points[next].cluster != NOT_CLASSIFIED) continue;
            dfs(next, c);
        }
    }
    
    void checkNearPoints() {
        for(int i=0;i<size;i++) {
            for(int j=0;j<size;j++) {
                if(i==j) continue;
                if(points[i].getDis(points[j]) <= eps) {
                    points[i].ptsCnt++;
                    adjPoints[i].push_back(j);
                }
            }
        }
    }
    // is idx'th point core object?
    bool isCoreObject(int idx) {
        // return true;
        return points[idx].ptsCnt >= minPts;
    }
    
    std::vector<std::vector<int> > getCluster() {
        return cluster;
    }
};

std::vector<std::vector<int> > dbscan_entry(int num_pts, double eps, int min_pts, 
                                            std::vector<LidarPoint> &lidarpts)
{
    std::vector<Point> vec_pts;
    for (auto lpt : lidarpts)
        vec_pts.push_back({lpt.rx, lpt.ry, 0, NOT_CLASSIFIED});
    printf("input ok\n");
    
    DBSCAN dbScan(num_pts, eps, min_pts, vec_pts);
    printf("class init ok\n");

    dbScan.run();
    printf("class run ok\n");

    return dbScan.getCluster();
}
