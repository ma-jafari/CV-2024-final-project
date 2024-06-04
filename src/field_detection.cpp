#include "field_detection.hpp"
#include "opencv2/core/mat.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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
                    const cv::Mat &mask) {
  //  NOTE: 1.3f instead of 1.0f to give some leeway for not perfectly aligned
  //  contours
  constexpr float hough_leeway = 1.3f;
  constexpr int hough_threshold = 300;
  vector<Vec2f> lines;
  lines.reserve(20);
  HoughLines(gray_contours, lines, hough_leeway, CV_PI / 180, hough_threshold);
  if (cluster_lines) {
    constexpr int n_clusters = 10;
    vector<Vec2f> centers;
    Mat labels;
    kmeans(lines, MIN(n_clusters, lines.size()), labels,
           TermCriteria(TermCriteria::EPS, 10, 1.0), 3, KMEANS_PP_CENTERS,
           centers);
    lines = centers;
  }

  for (size_t i = 0; i < lines.size(); i++) {
    float rho = lines[i][0], theta = lines[i][1];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a * rho, y0 = b * rho;
    pt1.x = cvRound(x0 + 1000 * (-b));
    pt1.y = cvRound(y0 + 1000 * (a));
    pt2.x = cvRound(x0 - 1000 * (-b));
    pt2.y = cvRound(y0 - 1000 * (a));
    line(mask, pt1, pt2, Scalar(255), 3, LINE_AA);
  }
}

} // namespace
cv::Mat detect_field(const cv::Mat &input_image) {
  using namespace cv;
  using namespace std;
  Mat in = input_image.clone();
  // imshow("original", in);
  medianBlur(in, in, 7);                // to reduce noise
  in -= Scalar(255, 0, 255);            // remove blue and red components
  threshold(in, in, 100, 255, CV_8UC1); // to reduce noise
  // imshow("removed blue", in);

  Mat graycontours;
  get_field_contours(in, graycontours);

  Mat mask = Mat::zeros(in.rows + 2, in.cols + 2, CV_8U);
  get_field_mask(graycontours, false, mask);
  // imshow("Mask", mask);
  Mat filled = Mat::zeros(in.rows, in.cols, in.type());
  Scalar newVal = Scalar(255);

  // FIX: Use image moments to estimate center of pool table
  int area = floodFill(filled, mask, Point(in.cols / 2, in.rows / 2),
                       Scalar(0, 255, 255), nullptr, Scalar(), Scalar());
  // imshow("Segmented pool table", filled);
  // waitKey(0);
  return filled;
}
