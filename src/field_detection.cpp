#include "field_detection.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <strings.h>
#include <vector>

#define PRINTS

cv::Scalar colori[4] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
                        cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255)};
namespace {

using namespace cv;
using namespace std;
void get_field_contours(const cv::Mat &in, cv::Mat &gray_contours) {
  constexpr float canny_thresh = 80;
  RNG rng(42);
  Mat canny_output;
  Canny(in, canny_output, canny_thresh, canny_thresh * 2);
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(canny_output, contours, hierarchy, RETR_TREE,
               CHAIN_APPROX_SIMPLE);
  Mat countours = Mat::zeros(canny_output.size(), CV_8UC3);
  for (size_t i = 0; i < contours.size(); i++) {
    Scalar color =
        Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    drawContours(countours, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
  }
  // imshow("Contours", countours);
  cvtColor(countours, gray_contours, COLOR_BGR2GRAY);
}
void get_field_mask(const cv::Mat &gray_contours, bool cluster_lines,
                    const cv::Mat &mask, vector<Vec2f> &lines) {
  //  NOTE: 1.3f instead of 1.0f to give some leeway for not perfectly aligned
  //  contours
  constexpr float hough_leeway = 1.3f;
  constexpr int hough_threshold = 300;
  lines.reserve(50);
  HoughLines(gray_contours, lines, hough_leeway, CV_PI / 180, hough_threshold);
  if (cluster_lines) {
    constexpr int n_clusters = 10;
    vector<Vec2f> centers;
    centers.reserve(MIN(n_clusters, lines.size()));
    Mat labels;
    kmeans(lines, MIN(n_clusters, lines.size()), labels,
           TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_PP_CENTERS,
           centers);
    lines = centers;
  }

#ifdef PRINTS
  float rho_mean = 0.0f;
  float theta_mean = 0.0f;
  for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    rho_mean += rho;
    theta_mean += theta;
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(mask, pt1, pt2, Scalar(255), 3, LINE_AA);
  }
#endif
}

void get_line_clusters(const vector<float> &vec, vector<int> &labels) {
  labels.clear();
  constexpr int n_clusters = 2;
  kmeans(vec, n_clusters, labels, TermCriteria(TermCriteria::EPS, 10, 1.0), 3,
         KMEANS_PP_CENTERS, noArray());
}

int get_strictest_line(const vector<float> &rhos, const vector<float> &thetas,
                       const Point image_center) {
  float min_dist = numeric_limits<float>::max();
  int index_closest_line = -1;
  for (int i = 0; i < rhos.size(); ++i) {
    float a, b, c;
    a = cos(thetas[i]), b = sin(thetas[i]);
    c = -rhos[i];
    // NOTE: since ||(a,b)||=sqrt(sin(theta)^2+cos(theta)^2) = 1 for all theta
    // I dont need to divide by this norm when computing the distance
    float dist = fabs(a * image_center.x + b * image_center.y + c);
    if (dist < min_dist) {
      min_dist = dist;
      index_closest_line = i;
    }
  }
  return index_closest_line;
}
} // namespace

void detect_field(const cv::Mat &input_image) {
  using namespace cv;
  using namespace std;
  Mat in = input_image.clone();
  // imshow("original", in);
  // PERF: median blur is very slow, is there any other way??
  // FIX: Try to find a better way
  medianBlur(in, in, 7);                // to reduce noise
  in -= Scalar(255, 0, 255);            // remove blue and red components
  threshold(in, in, 100, 255, CV_8UC1); // to reduce noise
  // imshow("removed blue", in);

  Mat graycontours;
  get_field_contours(in, graycontours);

  Mat mask = Mat::zeros(in.rows + 2, in.cols + 2, CV_8U);
  vector<Vec2f> lines;
  vector<float> thetas;
  get_field_mask(graycontours, false, mask, lines);
  // TODO:

  for (auto el : lines) {
    thetas.push_back(el[1] < CV_PI / 2 ? CV_PI - el[1] : el[1]);
  }
  constexpr int n_clusters = 2;
  vector<int> labels;
  get_line_clusters(thetas, labels);

#ifdef PRINTS

  Mat color_lines2 = Mat::zeros(in.rows, in.cols, in.type());
  ;

  for (size_t i = 0; i < thetas.size(); i++) {

    float rho = lines[i][0], theta = lines[i][1];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(color_lines2, pt1, pt2, colori[1 + labels[i]], 3, LINE_AA);
  }
  imshow("first subdivision", color_lines2);
#endif
  // for (auto l : labels) {
  //   cout << l << endl;
  // }
  vector<float> groupA;
  vector<float> groupA_thetas;
  vector<float> groupB;
  vector<float> groupB_thetas;
  for (int i = 0; i < lines.size(); ++i) {
    if (labels[i] == 0) {
      groupA.push_back(lines[i][0]);
      groupA_thetas.push_back(lines[i][1]);
    } else {
      groupB.push_back(lines[i][0]);
      groupB_thetas.push_back(lines[i][1]);
    }
  }
  //  cout << "A:" << groupA.size() << endl;
  //  cout << "B:" << groupB.size() << endl;

  Mat color_lines = Mat::zeros(in.rows, in.cols, in.type());
  get_line_clusters(groupA, labels);

  vector<float> groupC;
  vector<float> groupC_thetas;
  vector<float> groupD;
  vector<float> groupD_thetas;
  for (int i = 0; i < groupA.size(); ++i) {
    cout << labels[i] << endl;
    if (labels[i] == 0) {
      groupC.push_back(groupA[i]);
      groupC_thetas.push_back(groupA_thetas[i]);
    } else {
      groupD.push_back(groupA[i]);
      groupD_thetas.push_back(groupA_thetas[i]);
    }
  }
  cout << "size group C" << groupC_thetas.size() << endl;
#ifdef PRINTS
  for (size_t i = 0; i < groupA.size(); i++) {
    float rho = groupA[i], theta = groupA_thetas[i];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    cout << labels[i] << endl;
    line(color_lines, pt1, pt2, colori[2 + labels[i]], 3, LINE_AA);
  }
#endif

  const Point image_center(color_lines.cols / 2, color_lines.rows / 2);
  circle(color_lines, image_center, 10, Scalar(255, 0, 255));

  int index_closest_line =
      get_strictest_line(groupC, groupC_thetas, image_center);
  float rho = groupC[index_closest_line],
        theta = groupC_thetas[index_closest_line];
  Point pt1, pt2;
  double a = cos(theta), b = sin(theta);
  cout << "a " << a << endl;
  cout << "b " << b << endl;
  cout << "rho " << rho << endl;
  cout << "---------------------------------" << endl;
  double x0 = a * rho, y0 = b * rho;
  pt1.x = cvRound(x0 + 1000 * (-b));
  pt1.y = cvRound(y0 + 1000 * (a));
  pt2.x = cvRound(x0 - 1000 * (-b));
  pt2.y = cvRound(y0 - 1000 * (a));
  line(color_lines, pt1, pt2, Scalar(255, 0, 255), 3, LINE_AA);

  ////////////////////////

  index_closest_line = get_strictest_line(groupD, groupD_thetas, image_center);
  rho = groupD[index_closest_line], theta = groupD_thetas[index_closest_line];
  pt1, pt2;
  a = cos(theta), b = sin(theta);
  cout << "a " << a << endl;
  cout << "b " << b << endl;
  cout << "rho " << rho << endl;
  cout << "---------------------------------" << endl;
  x0 = a * rho, y0 = b * rho;
  pt1.x = cvRound(x0 + 1000 * (-b));
  pt1.y = cvRound(y0 + 1000 * (a));
  pt2.x = cvRound(x0 - 1000 * (-b));
  pt2.y = cvRound(y0 - 1000 * (a));
  line(color_lines, pt1, pt2, Scalar(255, 0, 255), 3, LINE_AA);
  ////HROUP B
  get_line_clusters(groupB, labels);

#ifdef PRINTS
  for (size_t i = 0; i < groupB.size(); i++) {
    float rho = groupB[i], theta = groupB_thetas[i];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(color_lines, pt1, pt2, colori[labels[i]], 3, LINE_AA);
  }
#endif

  cout << "########################################" << endl;

#ifdef PRINTS
#endif
  imshow("cc", color_lines);
  waitKey(0);
}
