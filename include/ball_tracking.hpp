/*Author: Matteo De Gobbi */
#ifndef BallTracking
#define BallTracking

#include "ball_classification.hpp"
#include "field_detection.hpp"
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

void track_balls(std::string path, std::vector<cv::Rect> &bboxes,
                 std::vector<ball_class> &ball_classes, bool savevideo,
                 std::string out_savepath, Vec4Points table_vertices);

#endif // !BallTracking
