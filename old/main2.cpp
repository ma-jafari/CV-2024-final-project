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
    //    Mat final = in.clone();
    Mat final = Mat::zeros(in.rows, in.cols, CV_8U);
    Mat gray_sharpened;
    cvtColor(in, in, COLOR_BGR2HSV);
    GaussianBlur(in, in, Size(5, 5), 2);
    Laplacian(in, laplacian, CV_16S);
    in -= 5 * laplacian;
    Mat hsv_channels[3];
    split(in, hsv_channels);
    gray_sharpened = hsv_channels[2] + hsv_channels[0];
    imshow("sharp", gray_sharpened);
    Canny(gray_sharpened, gray_sharpened, 100, 500);

    imshow("cann", gray_sharpened);
    Mat backup = gray_sharpened.clone();

    Mat kernD = getStructuringElement(MORPH_RECT, Size(7, 7));
    Mat kernE = getStructuringElement(MORPH_RECT, Size(15, 15));
    // morphologyEx(out, out, MORPH_OPEN, kernD);
    dilate(gray_sharpened, gray_sharpened, kernD);
    erode(gray_sharpened, gray_sharpened, kernE);
    gray_sharpened = backup - gray_sharpened;
    imshow("after diler", gray_sharpened);
    // waitKey(0);

    // dilate(out, out, kern);
    // out = backup - out;
    // imshow("remove chairs", out);
    std::vector<Vec4i> lines;
    HoughLinesP(gray_sharpened, lines, 1, CV_PI / 180, 100);

    std::vector<Vec2f> coeffs;
    for (size_t i = 0; i < lines.size(); i++) {
      Vec4i l = lines[i];
      line(final, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 3,
           LINE_AA);
      float m = (float)(l[3] - l[1]) / (float)(l[2] - l[0] + 1e-7);
      float q = (float)l[1] - m * l[0];
      if (!isnan(m) && !isnan(q)) {
        coeffs.push_back(Vec2f(m, q));
        // cout << m << endl;
      }
    }

    vector<Point2f> newlines;
    HoughLines(final, newlines, 1, CV_PI / 180, 100);
    for (int i = 0; i < newlines.size(); i++) {
      float rho = newlines[i].x, theta = newlines[i].y;
      double m = cos(theta), b = sin(theta);
      Point pt1, pt2;
      double x0 = 0, y0 = b;
      pt1.x = cvRound(x0 + 1000 * (-b));
      pt1.y = cvRound(y0 + 1000 * (m));
      pt2.x = cvRound(x0 - 1000 * (-b));
      pt2.y = cvRound(y0 - 1000 * (m));

      line(final, pt1, pt2, Scalar(155), 3, LINE_AA);
    }
    /*
        for (int i = 0; i < coeffs.size(); i++) {
          float m = coeffs[i][0], b = coeffs[i][1];
          Point pt1, pt2;
          double x0 = 0, y0 = b;
          pt1.x = cvRound(x0);
          pt1.y = cvRound(m * x0 + b);
          pt2.x = cvRound(1000);
          pt2.y = cvRound(m * 1000 + b);
          if (1) {

            line(final, pt1, pt2, Scalar(255, 0, 255), 3, LINE_AA);
          }
        }
        */

    /*vector<Point2f> centers;
    Mat labels;
    cout << coeffs.size() << endl;
    kmeans(coeffs, MIN(4, coeffs.size()), labels,
           TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 20,
           KMEANS_PP_CENTERS, centers);

    for (int i = 0; i < centers.size(); i++) {
      float m = centers[i].x, b = centers[i].y;
      Point pt1, pt2;
      double x0 = 0, y0 = b;
      pt1.x = cvRound(x0);
      pt1.y = cvRound(m * x0 + b);
      pt2.x = cvRound(1000);
      pt2.y = cvRound(m * 1000 + b);
      if (1) {

        line(final, pt1, pt2, Scalar(255, 0, 255), 3, LINE_AA);
      }
    }*/
    imshow("detected lines", final);
    // imshow("finestra", im);
    waitKey(0);
  }
  return 0;
}
