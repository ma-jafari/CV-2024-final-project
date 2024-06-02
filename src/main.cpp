#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "field_detection.hpp"

int main() {
	std::cout << DetectField() << std::endl;
	cv::Mat im = cv::imread("data/game1_clip3/frames/frame_first.png");
	cv::imshow("finestra",im);
	cv::waitKey(0);
	return 0;
}
