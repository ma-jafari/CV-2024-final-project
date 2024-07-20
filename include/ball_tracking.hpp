/*Author: Matteo De Gobbi */
#ifndef BallTracking
#define BallTracking

#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

void track_balls(std::string path, std::vector<cv::Rect> bboxes, bool savevideo,
                 std::string out_savepath);

#endif // !BallTracking
