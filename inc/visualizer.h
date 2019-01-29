#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "global.h"
#include "object.h"

const float map_range_length = 100.0;
const float map_range_width = 50.0;
const float map_scale = 0.2;

class Visualizer
{
public:
    Visualizer();

    void Init();
    void DrawGT(const std::vector<BoxObject> &v);
    void DrawRadarObjs(const std::vector<RadarObject> &v);
    void ShowMap() const;

private:
    cv::Mat local_map;
};

#endif // VISUALIZER_H
