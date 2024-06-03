#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <strings.h>
#include <vector>

#include "field_detection.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"

int main() {
  using namespace cv;
  using namespace std;
  string names[] = {"game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4",
                    "game2_clip1", "game2_clip2", "game3_clip1", "game3_clip2",
                    "game4_clip1", "game4_clip2"};
  RNG rng(12);
  float thresh = 80;

  for (int index = 0; index < 10; index++) {
    Mat in = cv::imread(string("data/") + names[index] +
                        string("/frames/frame_first.png"));

    medianBlur(in, in, 7);
    in -= Scalar(255, 0, 255);
    threshold(in, in, 100, 255, CV_8UC1);
    imshow("removed blue", in);
    Mat canny_output;
    Canny(in, canny_output, thresh, thresh * 2);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, RETR_TREE,
                 CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++) {
      Scalar color =
          Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
      drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
    }
    Mat greycontours;
    cvtColor(drawing, greycontours, COLOR_BGR2GRAY);
    imshow("Contours", drawing);
    imshow("GContours", greycontours);
    Mat final = in.clone();
    vector<Vec2f> lines;
    // FIX: 1.3f to give some leeway for not perfectly aligned contours
    HoughLines(greycontours, lines, 1.3f, CV_PI / 180, 300);
    vector<Vec2f> centers;
    Mat labels;
    kmeans(lines, MIN(10, lines.size()), labels,
           TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_PP_CENTERS,
           centers);

    lines = centers;
    for (size_t i = 0; i < lines.size(); i++) {
      float rho = lines[i][0], theta = lines[i][1];
      Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a * rho, y0 = b * rho;
      pt1.x = cvRound(x0 + 1000 * (-b));
      pt1.y = cvRound(y0 + 1000 * (a));
      pt2.x = cvRound(x0 - 1000 * (-b));
      pt2.y = cvRound(y0 - 1000 * (a));
      line(final, pt1, pt2, Scalar(255, 0, 255), 3, LINE_AA);
    }
    imshow("hough", final);
    waitKey(0);
  }
  return 0;
}
