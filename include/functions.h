#pragma once

#include "field_detection.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Pixel {
  int x, y;
  Pixel(int x, int y) : x(x), y(y) {}
};

struct ball_detection_params {
  float min_Dist;
  int min_Radius;
  float TH_Circ_A;
  float TH_Circ_a;
  float TH_Circ_B;
  float TH_Ratio_B;
  float TH_Circ_C;
  float TH_Ratio_C;
};

/*--------------------------------------SPECIFIC
 * FUNCTIONS---------------------------------------------*/
void erode_image(const cv::Mat &current_image, cv::Mat &closed,
                 int erosion_size);
void erode_image(const cv::Mat &current_image, cv::Mat &dilated,
                 int dilation_size);
void get_circles(const cv::Mat &input_img, std::vector<cv::Vec3f> &circles,
                 float sensibility, ball_detection_params &ball_params);
void select_circles(std::vector<cv::Vec3f> &circles,
                    ball_detection_params &ball_params);
void extractFrames(const std::string &videoPath, int frameInterval);
std::vector<std::vector<cv::Point2f>>
compute_bbox_vertices(const std::vector<cv::Vec3f> &circless);
std::vector<std::vector<float>> calculate_predictedBoxes(
    const std::vector<std::vector<cv::Point2f>> &allVertices);
void draw_circles(std::vector<cv::Vec3f> &circles, const cv::Mat &image);
void draw_bboxes(const std::vector<std::vector<cv::Point2f>> &vertices,
                 cv::Mat &image);

float calculateArea(const std::vector<float> &box);
float calculateIntersection(const std::vector<float> &box1,
                            const std::vector<float> &box2);
float calculateIoU(const std::vector<float> &box1,
                   const std::vector<float> &box2);
float calculateAccuracy(const std::vector<std::vector<float>> &predictedBoxes,
                        const std::vector<std::vector<float>> &groundTruthBoxes,
                        float threshold);

std::vector<cv::Point2f> OMOGRAFIA(const std::vector<cv::Vec3f> circles,
                                   const Vec4Points vertices, int width,
                                   int height);
