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
    Mat final = in.clone();
    Mat gray_sharpened;
    cvtColor(in, in, COLOR_BGR2HSV);
    GaussianBlur(in, in, Size(7, 7), 3);
    Laplacian(in, laplacian, CV_16S);
    in -= 5 * laplacian;
    Mat hsv_channels[3];
    split(in, hsv_channels);
    cout << "assasd" << endl;
    gray_sharpened = hsv_channels[2];
    imshow("sharp", gray_sharpened);
    Canny(gray_sharpened, gray_sharpened, 200, 300);

    imshow("cann", gray_sharpened);
    Mat backup = gray_sharpened.clone();

    Mat kernD = getStructuringElement(MORPH_RECT, Size(7, 7));
    Mat kernE = getStructuringElement(MORPH_RECT, Size(15, 15));
    // morphologyEx(out, out, MORPH_OPEN, kernD);
    // dilate(gray_sharpened, gray_sharpened, kernD);
    // erode(gray_sharpened, gray_sharpened, kernE);
    // gray_sharpened = backup - gray_sharpened;
    imshow("after diler", gray_sharpened);
    // waitKey(0);

    // dilate(out, out, kern);
    // out = backup - out;
    // imshow("remove chairs", out);
    std::vector<Vec2f> lines;
    HoughLines(gray_sharpened, lines, 1, CV_PI / 180, 200);
    Mat labels;
    vector<Point2f> centers;
    kmeans(lines, MIN(4, lines.size()), labels,
           TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 20,
           KMEANS_PP_CENTERS, centers);
    cout << centers[0] << endl;
    cout << centers[1] << endl;
    cout << centers[2] << endl;
    cout << centers[3] << endl;
    for (int i = 0; i < centers.size(); i++) {
      float rho = centers[i].x, theta = centers[i].y;
      Point pt1, pt2;
      double m = cos(theta), b = sin(theta);
      double x0 = m * rho, y0 = b * rho;
      pt1.x = cvRound(x0 + 1000 * (-b));
      pt1.y = cvRound(y0 + 1000 * (m));
      pt2.x = cvRound(x0 - 1000 * (-b));
      pt2.y = cvRound(y0 - 1000 * (m));
      if (1 || theta < 5) {

        line(final, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
      }
    }
    for (int i = 0; false && i < lines.size(); i++) {
      float rho = lines[i][0], theta = lines[i][1];
      Point pt1, pt2;
      double m = cos(theta), b = sin(theta);
      double x0 = m * rho, y0 = b * rho;
      pt1.x = cvRound(x0 + 1000 * (-b));
      pt1.y = cvRound(y0 + 1000 * (m));
      pt2.x = cvRound(x0 - 1000 * (-b));
      pt2.y = cvRound(y0 - 1000 * (m));
      if (1 || theta < 5) {

        line(final, pt1, pt2, Scalar(255, 0, 255), 3, LINE_AA);
      }
    }
    imshow("detected lines", final);
    // imshow("finestra", im);
    waitKey(0);
  }
  return 0;
}
