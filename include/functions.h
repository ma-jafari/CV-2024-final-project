#pragma once

#include "field_detection.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Pixel {
  int x, y;
  Pixel(int x, int y) : x(x), y(y) {}
};
/*--------------------------------------SPECIFIC
 * FUNCTIONS---------------------------------------------*/
void EROSION(const cv::Mat &current_image, cv::Mat &closed, int erosion_size);
void DILATION(const cv::Mat &current_image, cv::Mat &dilated,
              int dilation_size);
void Hough_Circles(const cv::Mat &input_img, cv::Mat &img_with_selected_circles,
                   std::vector<cv::Vec3f> &circles, float min_Dist,
                   float sensibility, int min_Radius, float TH_Circ_A,
                   float TH_Circ_a, float TH_Circ_B, float th_Ratio_B,
                   float TH_Circ_C, float th_Ratio_C);
void select_Circles(std::vector<cv::Vec3f> &circles, float TH_Circ_A,
                    float TH_Circ_a, float TH_Circ_B, float th_Ratio_B,
                    float TH_Circ_C, float th_Ratio_C);
void extractFrames(const std::string &videoPath, int frameInterval);
std::vector<std::vector<cv::Point2f>>
calculate_SquaresVertices(const std::vector<cv::Vec3f> &circless);
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
