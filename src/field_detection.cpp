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

#define __PRINTS

// colors used for visualization of clustered lines
cv::Scalar line_colors[4] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
                             cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255)};

// anonymous namespace used to have functions local to this file
// and to avoid namespace pollution from the using statement
namespace {

using namespace cv;
using namespace std;
void noise_removal_preprocess(Mat &in, bool show_intermediate) {
  // NOTE: median blur is slower than gaussian blur
  // but it performs better
  medianBlur(in, in, 7);                // to reduce noise
  in -= Scalar(255, 0, 255);            // remove blue and red components
  threshold(in, in, 100, 255, CV_8UC1); // to reduce noise
  if (show_intermediate) {
    imshow("table segm:median blur+thresholding", in);
  }
  Mat kernel = getStructuringElement(MORPH_RECT, Size(13, 13));
  dilate(in, in, kernel);
  erode(in, in, kernel);
  if (show_intermediate) {
    imshow("table segm:closing morph operation", in);
  }
}
void get_field_contours(const Mat &in, Mat &gray_contours,
                        bool show_intemediate) {
  constexpr float canny_thresh = 60;
  RNG rng(42);
  Mat canny_output;
  Canny(in, canny_output, canny_thresh, canny_thresh * 2);
  if (show_intemediate) {
    imshow("table segm:Canny", canny_output);
  }
  gray_contours = canny_output;
  /* dilate(gray_contours, gray_contours,
          getStructuringElement(MORPH_RECT, Size(3, 3)));
   return;*/
  // TODO: CHANGE THIS PROBABLY
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
  imshow("contours", gray_contours);
}
void get_hough_lines(const cv::Mat &gray_contours, vector<Vec2f> &lines) {
  //  NOTE: 1.3f instead of 1.0f to give some leeway for detecting not perfectly
  //  aligned lines
  constexpr float hough_leeway = 1.3f;
  constexpr int hough_threshold = 300;
  lines.reserve(50); // To avoid reallocations
  HoughLines(gray_contours, lines, hough_leeway, CV_PI / 180, hough_threshold);
}

void get_line_clusters(const vector<float> &vec, vector<int> &labels) {
  labels.clear();
  constexpr int n_clusters = 2; // we always divide the lines in two clusters
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
  Point pt1, pt2;
  float a = cos(theta), b = sin(theta);
  float x0 = a * rho, y0 = b * rho;
  pt1.x = cvRound(x0 + 1000 * (-b));
  pt1.y = cvRound(y0 + 1000 * (a));
  pt2.x = cvRound(x0 - 1000 * (-b));
  pt2.y = cvRound(y0 - 1000 * (a));
  line(image, pt1, pt2, Scalar(255, 0, 255), 3, LINE_AA);
}

// offset colors is set to true if we want to use red and yellow instead of
// green and blue top differentiate the various clusters
void draw_cluster_lines(Mat &out, vector<float> rhos, vector<float> thetas,
                        vector<int> labels, bool offset_colors) {

  for (size_t i = 0; i < rhos.size(); i++) {
    float rho = rhos[i], theta = thetas[i];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(out, pt1, pt2, line_colors[2 * offset_colors + labels[i]], 3, LINE_AA);
  }
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
  bool show_intermediate = true;
  Mat in = input_image.clone();

  noise_removal_preprocess(in, show_intermediate);

  Mat graycontours;
  get_field_contours(in, graycontours, show_intermediate);

  vector<Vec2f> lines;
  vector<float> thetas;
  vector<float> rhos;
  get_hough_lines(graycontours, lines);

  // update thetas so that clustering is robust to camera projection, the reason
  // for this is explained in the report
  for (auto el : lines) {
    thetas.push_back(el[1] < CV_PI / 2 ? CV_PI - el[1] : el[1]);
  }
  vector<int> labels;
  get_line_clusters(thetas, labels); // cluster lines based on their theta
  vector<float> rhos_1;
  vector<float> thetas_1;
  vector<float> rhos_2;
  vector<float> thetas_2;

  // split lines based on the first clustering
  for (int i = 0; i < lines.size(); ++i) {
    if (labels[i] == 0) {
      rhos_1.push_back(lines[i][0]);
      thetas_1.push_back(lines[i][1]);
    } else {
      rhos_2.push_back(lines[i][0]);
      thetas_2.push_back(lines[i][1]);
    }
  }

  get_line_clusters(rhos_1, labels); // divide first cluster according to rho
  vector<float> rhos_A;
  vector<float> thetas_A;
  vector<float> rhos_B;
  vector<float> thetas_B;
  for (int i = 0; i < rhos_1.size(); ++i) { // split according to cluster
    if (labels[i] == 0) {
      rhos_A.push_back(rhos_1[i]);
      thetas_A.push_back(thetas_1[i]);
    } else {
      rhos_B.push_back(rhos_1[i]);
      thetas_B.push_back(thetas_1[i]);
    }
  }

  Mat color_lines = Mat::zeros(in.rows, in.cols, in.type());
  if (show_intermediate) {
    draw_cluster_lines(color_lines, rhos_1, thetas_1, labels, true);
  }

  get_line_clusters(rhos_2, labels); // divide second cluster according to rho
  vector<float> rhos_C;
  vector<float> thetas_C;
  vector<float> rhos_D;
  vector<float> thetas_D;
  for (int i = 0; i < rhos_2.size(); ++i) { // split according to cluster
    if (labels[i] == 0) {
      rhos_C.push_back(rhos_2[i]);
      thetas_C.push_back(thetas_2[i]);
    } else {
      rhos_D.push_back(rhos_2[i]);
      thetas_D.push_back(thetas_2[i]);
    }
  }
  if (show_intermediate) {
    draw_cluster_lines(color_lines, rhos_2, thetas_2, labels, false);
  }

  const Point image_center(in.cols / 2, in.rows / 2);
  int index_closest_line = get_strictest_line(rhos_A, thetas_A, image_center);
  float rhoA = rhos_A[index_closest_line],
        thetaA = thetas_A[index_closest_line];
  Vec2f lineA(rhoA, thetaA);

  index_closest_line = get_strictest_line(rhos_B, thetas_B, image_center);
  float rhoB = rhos_B[index_closest_line],
        thetaB = thetas_B[index_closest_line];
  Vec2f lineB(rhoB, thetaB);

  index_closest_line = get_strictest_line(rhos_C, thetas_C, image_center);
  float rhoC = rhos_C[index_closest_line],
        thetaC = thetas_C[index_closest_line];
  Vec2f lineC(rhoC, thetaC);

  index_closest_line = get_strictest_line(rhos_D, thetas_D, image_center);
  float rhoD = rhos_D[index_closest_line],
        thetaD = thetas_D[index_closest_line];
  Vec2f lineD(rhoD, thetaD);

  // find vertices of the table by intersecting the lines
  Vec2i AC = intersect_hough_lines(lineA, lineC);
  Vec2i AD = intersect_hough_lines(lineA, lineD);
  Vec2i BC = intersect_hough_lines(lineB, lineC);
  Vec2i BD = intersect_hough_lines(lineB, lineD);

  if (show_intermediate) { // show final clustering and final vertices
    Mat final_lines = in.clone();
    draw_hough_line(final_lines, rhoA, thetaA);
    draw_hough_line(final_lines, rhoB, thetaB);
    draw_hough_line(final_lines, rhoC, thetaC);
    draw_hough_line(final_lines, rhoD, thetaD);
    circle(final_lines, AC, 10, Scalar(255, 255, 10), LINE_8);
    circle(final_lines, AD, 10, Scalar(255, 255, 10), LINE_8);
    circle(final_lines, BC, 10, Scalar(255, 255, 10), LINE_8);
    circle(final_lines, BD, 10, Scalar(255, 255, 10), LINE_8);
    imshow("table segm:final clustering", color_lines);
    imshow("table segm:final lines", final_lines);
  }
  Vec4Points vertices(AC, BC, BD, AD);
  return vertices;
}
