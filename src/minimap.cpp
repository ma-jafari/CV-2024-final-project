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
void drawMinimap(const std::vector<cv::Rect> rectangles,
                 const Vec4Points vertices,
                 const std::vector<ball_class> ballsss) {
  int width = 700;
  int height = 350;
  std::string imagePath = "../table.png";

  // Ensure points are correctly initialized
  std::vector<cv::Point2f> points(vertices.val, vertices.val + 4);

  // calculate the longest and shortes lati
  float sideAC = calculateDistance(vertices[0], vertices[1]);
  float sideBC = calculateDistance(vertices[1], vertices[2]);
  float sideBD = calculateDistance(vertices[1], vertices[2]);
  float sideAD = calculateDistance(vertices[3], vertices[0]);

  // Creazione del vettore di coppie (lunghezza, coppia di punti)
  std::vector<std::pair<float, std::pair<cv::Point2f, cv::Point2f>>> sides = {
      {sideAC, {vertices[0], vertices[1]}},
      {sideBC, {vertices[1], vertices[2]}},
      {sideBD, {vertices[2], vertices[3]}},
      {sideAD, {vertices[3], vertices[0]}}};
  float average_shortest_before_sorting = (sides[0].first + sides[2].first) / 2;
  float average_longest_before_sorting = (sides[1].first + sides[3].first) / 2;

  // Ordinamento dei lati per lunghezza
  std::sort(sides.begin(), sides.end(), compareSides);
  float average_shortest_after_sorting = (sides[0].first + sides[1].first) / 2;
  float average_longest_after_sorting = (sides[2].first + sides[3].first) / 2;

  // CASO 1: LA FOTO ORIGINALE HA IL TAVOLO MESSO PER ORIZZONTALE (quindi stesso
  // orientamento del tavolo della minimappa)
  if (average_longest_before_sorting == average_longest_after_sorting) {
    // QUESTA PARTE SI PUò OMETTERE PER LOGICA MA VEDIAMO NELLA PRATICA SE
    // INVECE PUò SERVIRCI, OVVIAMENTE DA MODIFICARE IN CASO

    // uso il lato MENO LUNGO per determinar i vertici più in ALTO
    points[0] = sides[2].second.first;
    points[1] = sides[2].second.second;
    // uso il lato PIù LUNGO per determinar i vertici più in  BASSO
    points[2] = sides[3].second.first;
    points[3] = sides[3].second.second;
  } else { // CASO 2: LA FOTO ORIGINALE HA IL TAVOLO MESSO PER VERTICALE, quindi
           // switcho l ordine dei vertici

    // uso il lato PIù CORTO per determinar i vertici più in ALTO
    points[0] = sides[0].second.first;
    points[1] = sides[0].second.second;
    // uso il lato MENO CORTO per determinar i vertici più in BASSO
    points[2] = sides[1].second.second;
    points[3] = sides[1].second.first;
  }

  // Define destination points
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

  // Apply perspective transformation
  std::vector<cv::Point2f> mapped_points;
  cv::perspectiveTransform(points_to_map, mapped_points, homography_matrix);

  // carico immagine
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  cv::Size size(width, height);
  cv::resize(image, image, size);
  // disegniamo le palle
  for (int i = 0; i < rectangles.size(); i++) {

    ball_class label = ballsss[i];

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
  // disegno le traiettoria delle palle con cerchi pieni di raggio 1
  for (int i = 0; i < rectangles.size(); i++) {
    cv::circle(image, mapped_points[i], 1, cv::Scalar(255, 0, 255), -1);
  }
  cv::imshow("minimap", image);
}
