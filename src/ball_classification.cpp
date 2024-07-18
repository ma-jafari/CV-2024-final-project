#include "ball_classification.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

ball_class classify_ball(const cv::Mat &img) {
  using namespace cv;
  constexpr int color_threshold_white = 110;
  Mat only_white_pixels;
  inRange(img,
          Scalar(color_threshold_white, color_threshold_white,
                 color_threshold_white),
          Scalar(255, 255, 255), only_white_pixels);

  /*namedWindow("thresholded");
  resizeWindow("thresholded", 400, 50);
  resizeWindow("after closing", 400, 50);
  */
  //imshow("thresholded", only_white_pixels);
  morphologyEx(only_white_pixels, only_white_pixels, MORPH_OPEN,
               getStructuringElement(MORPH_RECT, Size(3, 3)));

  //imshow("after closing", only_white_pixels);
  double white_pixels_percent =
      static_cast<double>(countNonZero(only_white_pixels)) /
      (only_white_pixels.rows * only_white_pixels.cols);

  constexpr int color_threshold_black = 10;
  Mat only_black_pixels;
  inRange(img,
          Scalar(color_threshold_black, color_threshold_black,
                 color_threshold_black),
          Scalar(255, 255, 255), only_black_pixels);

  double black_pixels_percent =
      1.0 - static_cast<double>(countNonZero(only_black_pixels)) /
                (only_black_pixels.rows * only_black_pixels.cols);
  // You can fine-tune this threshold
  ball_class label;
  if (black_pixels_percent > 0.10) {
    label = ball_class::EIGHT_BALL;
    std::cout << "EIGHT_BALL" << std::endl;
  } else if (white_pixels_percent > 0.15) {
    label = ball_class::CUE;
    std::cout << "CUE" << std::endl;
  } else if (white_pixels_percent > 0.03) {
    label = ball_class::STRIPED;
    std::cout << "STRIPED" << std::endl;
  } else {
    label = ball_class::SOLID;
    std::cout << "SOLID" << std::endl;
  }
  //waitKey();
  return label;
}

int ReturnBallClass(ball_class classOfBall) {
	if (classOfBall == ball_class::CUE) return 1;
	else if (classOfBall == ball_class::EIGHT_BALL) return 2;
	else if (classOfBall == ball_class::STRIPED) return 3;
	else if (classOfBall == ball_class::SOLID) return 4;
}
