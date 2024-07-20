#ifndef BALL_CLASSIFICATION
#define BALL_CLASSIFICATION

#include <opencv2/core/mat.hpp>
enum class ball_class { SOLID, STRIPED, CUE, EIGHT_BALL };
ball_class classify_ball(const cv::Mat &img);
ball_class int2ball_class(int i);
int ReturnBallClass(ball_class classOfBall);
#endif
