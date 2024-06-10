#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

/*void detect_balls(cv::Mat &masked_field) {
  using namespace cv;
  using namespace std;
  Mat original_masked = masked_field.clone();
  Mat gray_masked;
  cvtColor(masked_field, masked_field, COLOR_BGR2GRAY);
  masked_field.convertTo(gray_masked, CV_8U);
  // medianBlur(gray_masked, gray_masked, 7);
  Mat laplacian;
  Mat abs_laplacian;
  Laplacian(gray_masked, laplacian, CV_16S);
  cout << "aa" << endl;
  convertScaleAbs(laplacian, abs_laplacian);
  gray_masked -= 3 * abs_laplacian;
  Mat inverse_threshold;
  threshold(gray_masked, inverse_threshold, 150, 255, cv::THRESH_BINARY);
  dilate(inverse_threshold, inverse_threshold,
         getStructuringElement(MORPH_RECT, Size(5, 5)));

  imshow("inverse", inverse_threshold);
  threshold(gray_masked, gray_masked, 0, 255, THRESH_OTSU + cv::THRESH_BINARY);
  imshow("before_subtractionmasked", gray_masked);
  gray_masked -= inverse_threshold;

  imshow("masked", gray_masked);
  erode(gray_masked, gray_masked,
        getStructuringElement(MORPH_RECT, Size(3, 3)));
  Mat canny_out;
  float canny_trheshold = 100;
  Canny(gray_masked, canny_out, 180, canny_trheshold * 2);
  //  dilate(canny_out, canny_out, getStructuringElement(MORPH_RECT, Size(3,
  //  3)));
  imshow("can", canny_out);
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(canny_out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

  Mat drawing = Mat::zeros(masked_field.rows, masked_field.cols, CV_8UC3);
  vector<vector<Point>> good_contours;
  vector<Vec4i> good_hierarchy;
  for (auto contour : contours) {
    auto perimeter = arcLength(contour, false);
    cout << perimeter << endl;
    vector<Point> approx_poly;
    approxPolyDP(contour, approx_poly, 0.04 * perimeter, false);
    if (approx_poly.size() > 4) {
      good_contours.push_back(contour);
      RotatedRect rect_ellipse = fitEllipse(contour);
      Point2f center = rect_ellipse.center;
      Size2f size = rect_ellipse.size;
      float max_ray = max(size.width, size.height);
      float min_ray = min(size.width, size.height);
      if (fabs(size.width - size.height) < 0.5 * max_ray && max_ray < 50 &&
          min_ray > 8) {
        ellipse(original_masked, rect_ellipse, Scalar(255, 255, 255), 2);
      }
    }
  }

  RNG rng(12121);
  for (size_t i = 0; i < good_contours.size(); i++) {
    Scalar color =
        Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    drawContours(drawing, good_contours, (int)i, color, 2, LINE_8,
                 good_hierarchy, 0);
  }
  imshow("Contours", original_masked);
}*/

void detect_balls(cv::Mat &masked_field) {
  using namespace cv;
  using namespace std;
  Mat original_masked = masked_field.clone();
  Mat gray_masked;
  cvtColor(masked_field, masked_field, COLOR_BGR2GRAY);
  masked_field.convertTo(gray_masked, CV_8U);
  // medianBlur(gray_masked, gray_masked, 7);
  Mat laplacian;
  Mat abs_laplacian;
  Laplacian(gray_masked, laplacian, CV_16S);
  cout << "aa" << endl;
  convertScaleAbs(laplacian, abs_laplacian);
  gray_masked -= 3 * abs_laplacian;
  Mat inverse_threshold;
  threshold(gray_masked, inverse_threshold, 150, 255, cv::THRESH_BINARY);
  dilate(inverse_threshold, inverse_threshold,
         getStructuringElement(MORPH_RECT, Size(5, 5)));

  imshow("inverse", inverse_threshold);
  threshold(gray_masked, gray_masked, 10, 255, cv::THRESH_BINARY_INV);
  imshow("before_subtractionmasked", gray_masked);
  gray_masked += inverse_threshold;

  imshow("masked", gray_masked);
  Mat canny_out;
  float canny_trheshold = 100;
  Canny(gray_masked, canny_out, 180, canny_trheshold * 2);
  //  dilate(canny_out, canny_out, getStructuringElement(MORPH_RECT, Size(3,
  //  3)));
  imshow("can", canny_out);
  /*
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(canny_out, contours, hierarchy, RETR_TREE,
    CHAIN_APPROX_SIMPLE);

    Mat drawing = Mat::zeros(masked_field.rows, masked_field.cols, CV_8UC3);
    RNG rng(12121);
    for (size_t i = 0; i < contours.size(); i++) {
      Scalar color =
          Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
      drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
    }
    imshow("Contours", drawing);
    */
}
