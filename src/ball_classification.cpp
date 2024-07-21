/*Author: Matteo De Gobbi */
#include "ball_classification.hpp"
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

ball_class classify_ball(const cv::Mat &img) {
  using namespace cv;
  constexpr int color_threshold_white = 100;
  Mat only_white_pixels;
  inRange(img,
          Scalar(color_threshold_white, color_threshold_white,
                 color_threshold_white),
          Scalar(255, 255, 255), only_white_pixels);

  morphologyEx(only_white_pixels, only_white_pixels, MORPH_OPEN,
               getStructuringElement(MORPH_RECT, Size(3, 3)));

  // imshow("after closing", only_white_pixels);
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
  ball_class label;
  constexpr double black_percent_thresh = 0.19;
  constexpr double white_percent_thresh = 0.20;
  constexpr double striped_percent_thresh = 0.05;
  if (black_pixels_percent > black_percent_thresh) {
    label = ball_class::EIGHT_BALL;
    // std::cout << "EIGHT_BALL" << std::endl;
  } else if (white_pixels_percent > white_percent_thresh) {
    label = ball_class::CUE;
    // std::cout << "CUE" << std::endl;
  } else if (white_pixels_percent > striped_percent_thresh) {
    label = ball_class::STRIPED;
    // std::cout << "STRIPED" << std::endl;
  } else {
    label = ball_class::SOLID;
    // std::cout << "SOLID" << std::endl;
  }
  // waitKey();
  return label;
}

ball_class int2ball_class(int i) {
  switch (i) {
  case 1:
    return ball_class::CUE;
  case 2:
    return ball_class::EIGHT_BALL;
  case 3:
    return ball_class::SOLID;
  case 4:
    return ball_class::STRIPED;
  default:
    return ball_class::SOLID;
  }
}

// Converts ball_class into the appriopriate color
cv::Scalar ball_class2color(ball_class c) {
  switch (c) {
  case ball_class::CUE:
    return cv::Scalar(255, 255, 255);
  case ball_class::EIGHT_BALL:
    return cv::Scalar(0, 0, 0);
  case ball_class::SOLID:
    return cv::Scalar(0, 0, 255);
  case ball_class::STRIPED:
    return cv::Scalar(255, 0, 0);
  }
}

bool is_ball_near_line(cv::Point2f ball_pos, float radius, cv::Point2f pointA,
                       cv::Point2f pointB) {
  float temp = (pointB.y - pointA.y) * ball_pos.x -
               (pointB.x - pointA.x) * ball_pos.y + pointB.x * pointA.y -
               pointB.y * pointA.x;
  float distance = fabs(temp) / norm(pointB - pointA);
  return distance < 1.0f * radius;
}
