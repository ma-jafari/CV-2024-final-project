/* Alessandro Di Frenna */
#include "ball_classification.hpp"
#include "field_detection.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

float calculateDistance(const cv::Point2f &p1, const cv::Point2f &p2) {
  return std::sqrt((p2.x - p1.x) * (p2.x - p1.x) +
                   (p2.y - p1.y) * (p2.y - p1.y));
}
bool compareSides(
    const std::pair<float, std::pair<cv::Point2f, cv::Point2f>> &a,
    const std::pair<float, std::pair<cv::Point2f, cv::Point2f>> &b) {
  return a.first < b.first;
}

void rotateCounterclockwise(std::vector<cv::Point2f> &points) {
  if (points.size() != 4) {
    std::cerr << "The vector must contain exactly 4 points." << std::endl;
    return;
  }

  // Store the first point
  cv::Point2f first_point = points[0];

  // Shift all points one position to the left
  for (size_t i = 0; i < points.size() - 1; ++i) {
    points[i] = points[i + 1];
  }

  // Place the first point in the last position
  points[3] = first_point;
}

float calculateAngle(const cv::Point2f &reference, const cv::Point2f &point) {
  return std::atan2(point.y - reference.y, point.x - reference.x);
}
void reorderVerticesClockwise(Vec4Points &vertices) {
  std::vector<cv::Point2f> points(vertices.val, vertices.val + 4);

  // Find the top-left vertex
  auto topLeftIt =
      std::min_element(points.begin(), points.end(),
                       [](const cv::Point2f &a, const cv::Point2f &b) {
                         return (a.y < b.y) || (a.y == b.y && a.x < b.x);
                       });

  // Set the top-left vertex as the starting point
  cv::Point2f topLeft = *topLeftIt;
  points.erase(topLeftIt);
  points.insert(points.begin(), topLeft);

  // Sort the remaining points based on the angle relative to the top-left
  // vertex
  cv::Point2f reference = points[0];
  std::sort(points.begin() + 1, points.end(),
            [reference](const cv::Point2f &a, const cv::Point2f &b) {
              return calculateAngle(reference, a) <
                     calculateAngle(reference, b);
            });

  // Update the vertices with the reordered points
  for (size_t i = 0; i < points.size(); ++i) {
    vertices.val[i] = points[i];
  }
}
void drawMinimap(const std::vector<cv::Rect> rectangles, Vec4Points vertices,
                 const std::vector<ball_class> balls) {
  int width = 700;
  int height = 350;
  std::string imagePath = "../table.png";

  reorderVerticesClockwise(vertices);
  std::vector<cv::Point2f> points(vertices.val, vertices.val + 4);

  float sideAC = calculateDistance(vertices[0], vertices[1]);
  float sideBC = calculateDistance(vertices[1], vertices[2]);
  float sideBD = calculateDistance(vertices[1], vertices[2]);
  float sideAD = calculateDistance(vertices[3], vertices[0]);

  std::vector<float> sides = {sideAC, sideBC, sideBD, sideAD};

  float average_shortest = (sides[1] + sides[3]) / 2;
  float average_longest = (sides[0] + sides[2]) / 2;

  if (average_shortest > average_longest)
    rotateCounterclockwise(points);

  std::vector<cv::Point2f> dst_points(4);
  dst_points[0] = cv::Point2f(0, 0);
  dst_points[1] = cv::Point2f(width, 0);
  dst_points[2] = cv::Point2f(width, height);
  dst_points[3] = cv::Point2f(0, height);

  cv::Mat homography_matrix = cv::getPerspectiveTransform(points, dst_points);

  std::vector<cv::Point2f> points_to_map(rectangles.size());
  for (int i = 0; i < rectangles.size(); i++) {
    int min_x = rectangles[i].x;
    int min_y = rectangles[i].y;
    int width = rectangles[i].width;
    int height = rectangles[i].height;

    points_to_map[i] = cv::Point2f(min_x + width / 2, min_y + height / 2);
  }

  std::vector<cv::Point2f> mapped_points;
  cv::perspectiveTransform(points_to_map, mapped_points, homography_matrix);

  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  cv::Size size(width, height);
  cv::resize(image, image, size);
  for (int i = 0; i < rectangles.size(); i++) {

    ball_class label = balls[i];

    switch (label) {
    case ball_class::EIGHT_BALL:
      cv::circle(image, mapped_points[i], 10, cv::Scalar(0, 0, 0), -1);
      break;
    case ball_class::CUE:
      cv::circle(image, mapped_points[i], 10, cv::Scalar(255, 255, 255), -1);
      break;
    case ball_class::STRIPED:
      cv::circle(image, mapped_points[i], 10, cv::Scalar(0, 0, 255), -1);
      break;
    case ball_class::SOLID:
      cv::circle(image, mapped_points[i], 10, cv::Scalar(255, 0, 0), -1);
      break;
    }
  }
  for (int i = 0; i < rectangles.size(); i++) {
    cv::circle(image, mapped_points[i], 1, cv::Scalar(255, 0, 255), -1);
  }
  cv::imshow("Image", image);
  cv::waitKey(0);
}
