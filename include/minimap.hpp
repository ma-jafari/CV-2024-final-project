#include "ball_classification.hpp"
#include "field_detection.hpp"
#include <opencv2/core.hpp>

void drawMinimap(const std::vector<cv::Rect> &rectangles, Vec4Points vertices,
                 const std::vector<ball_class> &balls, cv::Mat &minimap,
                 cv::Mat &trailmap, int minimap_w, int minimap_h);
