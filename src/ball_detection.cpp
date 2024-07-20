// Alessandro Di Frenna
#include "ball_detection.hpp"
#include "field_detection.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

/*--------------------------------------SPECIFIC
 * FUNCTIONS---------------------------------------------*/
void erode_image(const cv::Mat &current_image, cv::Mat &closed,
                 int erosion_size) {

  cv::Mat erosion_element = cv::getStructuringElement(
      cv::MORPH_CROSS, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
      cv::Point(erosion_size, erosion_size));

  cv::erode(current_image, closed, erosion_element);
}
void dilate_image(const cv::Mat &current_image, cv::Mat &dilated,
                  int dilation_size) {

  cv::Mat dilation_element = cv::getStructuringElement(
      cv::MORPH_CROSS, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
      cv::Point(dilation_size, dilation_size));

  cv::dilate(current_image, dilated, dilation_element);
}
void get_circles(const cv::Mat &input_img, vector<cv::Vec3f> &circles,
                 float sensibility, ball_detection_params &ball_params) {
  cv::Mat gray;
  if (input_img.channels() != 1)
    cv::cvtColor(input_img, gray, cv::COLOR_BGR2GRAY);
  else
    gray = input_img.clone();

  cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);
  cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, ball_params.min_Dist,
                   100, sensibility, ball_params.min_Radius, 16);
}

void select_circles(vector<cv::Vec3f> &circles,
                    ball_detection_params &ball_params) {
  int n_balls = circles.size();
  int new_size = circles.size();
  vector<bool> is_selected(n_balls, true);

  for (int i = 0; i < n_balls; i++) {
    for (int j = i + 1; j < n_balls; j++) {
      float x_first = circles[i][0], x_second = circles[j][0];
      float y_first = circles[i][1], y_second = circles[j][1];
      float r_first = circles[i][2], r_second = circles[j][2];

      float min = r_first < r_second ? r_first : r_second;
      float max = r_first >= r_second ? r_first : r_second;
      float ratio = min / max;

      float distance_Centers =
          sqrt(pow((x_second - x_first), 2) + pow((y_second - y_first), 2));
      float distance_Circ = distance_Centers - (r_first + r_second);

      if (r_first < 10 && r_second < 10) {

        if (distance_Circ < ball_params.TH_Circ_a) {
          is_selected[r_first < r_second ? i : j] = false;
          new_size--;
        } else {
          if (distance_Circ < 2 &&
              ratio < 0.91) { // NON ABBASSARE QUESTI 2 THRESHOLD: 1 , 0.91
            is_selected[r_first < r_second ? i : j] = false;
            new_size--;
          } else if (distance_Circ < 6 &&
                     ratio <
                         0.75) { // NON ABBASSARE QUESTI 2 THRESHOLD: 6 , 0.75
            is_selected[r_first < r_second ? i : j] = false;
            new_size--;
          }
        }
      } else {
        if (distance_Circ < ball_params.TH_Circ_A) {
          is_selected[r_first < r_second ? i : j] = false;
          new_size--;
        } else {
          if (distance_Circ < ball_params.TH_Circ_B &&
              ratio < ball_params.TH_Ratio_B) { // due threshold per vedere se
                                                // prende rumori
            is_selected[r_first < r_second ? i : j] = false;
            new_size--;
          } else if (distance_Circ < ball_params.TH_Circ_C &&
                     ratio < ball_params.TH_Ratio_C) { // due threshold per
                                                       // vedere se prende
                                                       // rumori
            is_selected[r_first < r_second ? i : j] = false;
            new_size--;
          }
        }
      }
    }
  }
  vector<cv::Vec3f> selected_circles;
  for (int i = 0; i < n_balls; ++i) {
    if (is_selected[i]) {
      selected_circles.push_back(circles[i]);
    }
  }
  /*
  int j = 0;
  vector<cv::Vec3f> selected_circles(new_size);
  for (int i = 0; i < n_balls; ++i) {
      if (is_selected[i]) {
          selected_circles[j] = circles[i];
          j++;
      }
  }*/

  circles = selected_circles;
}
void draw_circles(vector<cv::Vec3f> &circles, const cv::Mat &image) {

  for (int j = 0; j < circles.size(); j++) {
    cv::Vec3f c = circles[j];
    cv::Point center = cv::Point(c[0], c[1]);
    cv::circle(image, center, 1, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
    int radius = c[2];
    cv::circle(image, center, radius, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
  }
  cv::imshow("immagine FINALE cerchiata", image);
  cv::waitKey(0);
}
void draw_bboxes(const vector<vector<cv::Point2f>> &vertices, cv::Mat &image) {

  for (int i = 0; i < vertices.size(); i++) {

    vector<cv::Point2f> vec = vertices[i];

    cv::Point2f vertice1(vec[0]);
    cv::Point2f vertice2(vec[1]);
    cv::Point2f vertice3(vec[2]);
    cv::Point2f vertice4(vec[3]);

    cv::line(image, vertice1, vertice2, cv::Scalar(255, 0, 255), 2);
    cv::line(image, vertice2, vertice3, cv::Scalar(255, 0, 255), 2);
    cv::line(image, vertice3, vertice4, cv::Scalar(255, 0, 255), 2);
    cv::line(image, vertice4, vertice1, cv::Scalar(255, 0, 255), 2);
  }

  cv::imshow("immagine FINALE con boxes", image);
}

vector<vector<cv::Point2f>>
compute_bbox_vertices(const vector<cv::Vec3f> &circless) {

  vector<vector<cv::Point2f>> Allvertices(circless.size());

  for (int i = 0; i < circless.size(); i++) {
    cv::Vec3f circle = circless[i];

    float cx = circle[0];
    float cy = circle[1];
    float radius = circle[2];

    vector<cv::Point2f> vertices(4);
    vertices[0] =
        cv::Point2f(cx - radius, cy - radius); // Vertice superiore sinistro
    vertices[1] =
        cv::Point2f(cx + radius, cy - radius); // Vertice superiore destro
    vertices[2] =
        cv::Point2f(cx + radius, cy + radius); // Vertice inferiore destro
    vertices[3] =
        cv::Point2f(cx - radius, cy + radius); // Vertice inferiore sinistro

    Allvertices[i] = vertices;
  }

  return Allvertices;
}
vector<vector<float>>
compute_bboxes(const vector<vector<cv::Point2f>> &allVertices) {
  int n = allVertices.size();
  vector<vector<float>> predictedBoxes(n);

  for (int i = 0; i < n; i++) {
    vector<float> box(4);

    box[0] = allVertices[i][0].x; // x_min
    box[1] = allVertices[i][0].y; // y_min
    box[2] = allVertices[i][2].x; // x_max
    box[3] = allVertices[i][2].y; // y_max

    predictedBoxes[i] = box;
  }

  return predictedBoxes;
}

std::vector<cv::Vec3f> get_balls(cv::Mat &in_img) {

  int kernel_DILATION = 3;
  int kernel_EROSION = 3;
  float precisione_DIL = 13; // 12;
  float precisione_ERO = 11.5;

  ball_detection_params ball_params;
  ball_params.min_Dist = 2;
  ball_params.min_Radius = 8;
  ball_params.TH_Circ_A = -6;
  ball_params.TH_Circ_a = -4;
  ball_params.TH_Circ_B = 4;
  ball_params.TH_Ratio_B = 0.75;
  ball_params.TH_Circ_C = 8;
  ball_params.TH_Ratio_C = 0.6;

  cv::Mat dilated;
  cv::Mat eroded;
  erode_image(in_img, dilated, kernel_DILATION);

  erode_image(in_img, eroded, kernel_EROSION);
  /*
  imshow("dilated", dilated);
  imshow("eroded", eroded);
  */

  cv::Mat dilated_canny, eroded_canny;
  Canny(dilated, dilated_canny, 300, 300);
  Canny(eroded, eroded_canny, 300, 300);
  // imshow("dilcanny", dilated_canny);
  // imshow("eroded canny", eroded_canny);
  dilated = dilated_canny;
  eroded = eroded_canny;

  // dilated circle detection
  vector<cv::Vec3f> circles_dilated;
  get_circles(dilated, circles_dilated, precisione_DIL, ball_params);

  // eroded circle detection
  cv::Mat circle_EROSION;
  vector<cv::Vec3f> circles_erosion;
  get_circles(eroded, circles_erosion, precisione_ERO, ball_params);

  vector<cv::Vec3f> dil = circles_dilated;
  vector<cv::Vec3f> total = circles_erosion;

  total.insert(total.end(), dil.begin(), dil.end());

  select_circles(total, ball_params);

  return total;
}
