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
  for (int index = 0; index < 10; index++) {
    Mat in = cv::imread(string("data/") + names[index] +
                        string("/frames/frame_first.png"));
    Mat laplacian;
    //    Mat final = Mat::zeros(in.rows, in.cols, CV_8U);
    Mat final = in.clone();
    Mat gray_sharpened;
    cvtColor(in, in, COLOR_BGR2HSV);
    GaussianBlur(in, in, Size(1, 5), 2);
    // medianBlur(in, in, 2);
    Laplacian(in, laplacian, CV_16S);
    in -= 3 * laplacian;
    Mat hsv_channels[3];
    split(in, hsv_channels);
    gray_sharpened = hsv_channels[2]; // + hsv_channels[1] + hsv_channels[0];
    imshow("sharp", gray_sharpened);
    Canny(gray_sharpened, gray_sharpened, 150, 170);
    imshow("befire dilate", gray_sharpened);
    // dilate(gray_sharpened, gray_sharpened,
    //        getStructuringElement(MORPH_RECT, Size(3, 1)));
    vector<Vec2f> lines;
    HoughLines(gray_sharpened, lines, 2, CV_PI / 180 * 1.1, 330);
    // FIX:
    vector<Vec2f> centers;
    Mat labels;
    kmeans(lines, MIN(30, lines.size()), labels,
           TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_PP_CENTERS,
           centers);

    //   lines = centers;
    for (size_t i = 0; i < lines.size(); i++) {
      float rho = lines[i][0], theta = lines[i][1];
      Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a * rho, y0 = b * rho;
      pt1.x = cvRound(x0 + 1000 * (-b));
      pt1.y = cvRound(y0 + 1000 * (a));
      pt2.x = cvRound(x0 - 1000 * (-b));
      pt2.y = cvRound(y0 - 1000 * (a));
      line(final, pt1, pt2, Scalar(0, 255), 3, LINE_AA);
    }
    imshow("cann", gray_sharpened);
    imshow("lines", final);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(final, contours, hierarchy, RETR_FLOODFILL,
                 CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(in.size(), CV_8UC3);
    RNG rng(12345);
    for (size_t i = 0; i < contours.size(); i++) {
      Scalar color =
          Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
      drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
    }
    imshow("Contours", drawing);
    waitKey(0);
  }
  return 0;
}
