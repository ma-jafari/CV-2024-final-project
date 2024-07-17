#include "field_detection.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/matx.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#define NO_PRINTS

cv::Scalar colori[4] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
                        cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255)};
namespace {

using namespace cv;
using namespace std;
void get_field_contours(const cv::Mat &in, cv::Mat &gray_contours) {
  constexpr float canny_thresh = 60;
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
void get_hough_lines(const cv::Mat &gray_contours, bool cluster_lines,
                     vector<Vec2f> &lines) {
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
void draw_hough_line(Mat &image, float rho, float theta) {
#ifdef PRINTS
  Point pt1, pt2;
  float a = cos(theta), b = sin(theta);
  float x0 = a * rho, y0 = b * rho;
  pt1.x = cvRound(x0 + 1000 * (-b));
  pt1.y = cvRound(y0 + 1000 * (a));
  pt2.x = cvRound(x0 - 1000 * (-b));
  pt2.y = cvRound(y0 - 1000 * (a));
  line(image, pt1, pt2, Scalar(255, 0, 255), 3, LINE_AA);
#endif
}
Vec2i intersect_hough_lines(Vec2f line1, Vec2f line2) {

  cv::Mat A = (Mat_<double>(2, 2) << cos(line1[1]), sin(line1[1]),
               cos(line2[1]), sin(line2[1]));
  cv::Mat b = (Mat_<double>(2, 1) << line1[0], line2[0]);

  cv::Mat x;
  cv::solve(A, b, x);

  int x0 = static_cast<int>(std::round(x.at<double>(0, 0)));
  int y0 = static_cast<int>(std::round(x.at<double>(1, 0)));

  return Vec2i(x0, y0);
}
} // namespace

Vec4Points detect_field(const cv::Mat &input_image) {
  using namespace cv;
  using namespace std;
  Mat in = input_image.clone();
  // imshow("original", in);
  // PERF: median blur is very slow, is there any other way??
  // FIX: Try to find a better way
  medianBlur(in, in, 7);                // to reduce noise
  in -= Scalar(255, 0, 255);            // remove blue and red components
  threshold(in, in, 100, 255, CV_8UC1); // to reduce noise
  Mat kernel = getStructuringElement(MORPH_RECT, Size(13, 13));
  dilate(in, in, kernel);
  erode(in, in, kernel);

  // imshow("removed blue", in);

  Mat graycontours;
  get_field_contours(in, graycontours);

  vector<Vec2f> lines;
  vector<float> thetas;
  vector<float> rhos;
  get_hough_lines(graycontours, false, lines);

  for (auto el : lines) {
    thetas.push_back(el[1] < CV_PI / 2 ? CV_PI - el[1] : el[1]);
  }
  constexpr int n_clusters = 2;
  vector<int> labels;
  get_line_clusters(thetas, labels);
  vector<float> rhos_1;
  vector<float> thetas_1;
  vector<float> rhos_2;
  vector<float> thetas_2;

  for (int i = 0; i < lines.size(); ++i) {
    if (labels[i] == 0) {
      rhos_1.push_back(lines[i][0]);
      thetas_1.push_back(lines[i][1]);
    } else {
      rhos_2.push_back(lines[i][0]);
      thetas_2.push_back(lines[i][1]);
    }
  }
#ifdef PRINTS
  // FIRST SUBDIVISION
  Mat color_lines2 = Mat::zeros(in.rows, in.cols, in.type());
  for (size_t i = 0; i < thetas.size(); ++i) {
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

  Mat color_lines = Mat::zeros(in.rows, in.cols, in.type());
  const Point image_center(in.cols / 2, in.rows / 2);

  get_line_clusters(rhos_1, labels);
  vector<float> rhos_A;
  vector<float> thetas_A;
  vector<float> rhos_B;
  vector<float> thetas_B;
  for (int i = 0; i < rhos_1.size(); ++i) {
    if (labels[i] == 0) {
      rhos_A.push_back(rhos_1[i]);
      thetas_A.push_back(thetas_1[i]);
    } else {
      rhos_B.push_back(rhos_1[i]);
      thetas_B.push_back(thetas_1[i]);
    }
  }

#ifdef PRINTS
  // A AND B
  for (size_t i = 0; i < rhos_1.size(); i++) {
    float rho = rhos_1[i], theta = thetas_1[i];
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
  circle(color_lines, image_center, 10, Scalar(255, 0, 255));
#endif

  get_line_clusters(rhos_2, labels);
  vector<float> rhos_C;
  vector<float> thetas_C;
  vector<float> rhos_D;
  vector<float> thetas_D;
  for (int i = 0; i < rhos_2.size(); ++i) {
    if (labels[i] == 0) {
      rhos_C.push_back(rhos_2[i]);
      thetas_C.push_back(thetas_2[i]);
    } else {
      rhos_D.push_back(rhos_2[i]);
      thetas_D.push_back(thetas_2[i]);
    }
  }

#ifdef PRINTS
  for (size_t i = 0; i < rhos_2.size(); i++) {
    float rho = rhos_2[i], theta = thetas_2[i];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    cout << labels[i] << endl;
    line(color_lines, pt1, pt2, colori[labels[i]], 3, LINE_AA);
  }
  circle(color_lines, image_center, 10, Scalar(255, 0, 255));
#endif

  /*Mat final_lines =
      Mat::zeros(color_lines.rows, color_lines.cols, color_lines.type());
*/
  Mat final_lines = in.clone();
  int index_closest_line = get_strictest_line(rhos_A, thetas_A, image_center);
  float rho = rhos_A[index_closest_line], theta = thetas_A[index_closest_line];
  draw_hough_line(final_lines, rho, theta);
  Vec2f lineA(rho, theta);

  index_closest_line = get_strictest_line(rhos_B, thetas_B, image_center);
  rho = rhos_B[index_closest_line], theta = thetas_B[index_closest_line];
  draw_hough_line(final_lines, rho, theta);
  Vec2f lineB(rho, theta);

  index_closest_line = get_strictest_line(rhos_C, thetas_C, image_center);
  rho = rhos_C[index_closest_line], theta = thetas_C[index_closest_line];
  draw_hough_line(final_lines, rho, theta);
  Vec2f lineC(rho, theta);

  index_closest_line = get_strictest_line(rhos_D, thetas_D, image_center);
  rho = rhos_D[index_closest_line], theta = thetas_D[index_closest_line];
  draw_hough_line(final_lines, rho, theta);
  Vec2f lineD(rho, theta);

  Vec2i AC = intersect_hough_lines(lineA, lineC);
  circle(final_lines, AC, 10, Scalar(255, 255, 10), LINE_8);
  Vec2i AD = intersect_hough_lines(lineA, lineD);
  circle(final_lines, AD, 10, Scalar(255, 255, 10), LINE_8);
  Vec2i BC = intersect_hough_lines(lineB, lineC);
  circle(final_lines, BC, 10, Scalar(255, 255, 10), LINE_8);
  Vec2i BD = intersect_hough_lines(lineB, lineD);
  circle(final_lines, BD, 10, Scalar(255, 255, 10), LINE_8);
#ifdef PRINTS
  imshow("second clustering", color_lines);
  imshow("final lines", final_lines);
#endif
  Vec4Points vertices(AC, BC, BD, AD);
  return vertices;
}