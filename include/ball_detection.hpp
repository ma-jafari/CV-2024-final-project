#ifndef BallDetection
#define BallDetection

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace cv;
using namespace std;

bool isStriped(const Mat& img);
Mat detectBalls(const Mat& src);
std::vector<Vec3f> get_balls(cv::Mat& in_img);
bool is_ball_near_line(Point2f ball_pos, float radius, Point2f pointA, Point2f pointB);

#endif // !BallDetection
