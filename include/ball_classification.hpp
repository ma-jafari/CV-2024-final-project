#ifndef BALL_CLASSIFICATION
#define BALL_CLASSIFICATION

#include <opencv2/core/mat.hpp>
enum class ball_class { SOLID, STRIPED, CUE, EIGHT_BALL };
ball_class classify_ball(const cv::Mat &img);
#endif
