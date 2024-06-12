#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include <utility>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>
#include "field_detection.h"

using namespace std;
namespace fs = filesystem;

struct Pixel {
    int x, y;
    Pixel(int x, int y) : x(x), y(y) {}
};
/*--------------------------------------SPECIFIC FUNCTIONS---------------------------------------------*/
void EROSION(const cv::Mat& current_image,  cv::Mat& closed, int erosion_size);
void DILATION(const cv::Mat& current_image,  cv::Mat& dilated, int dilation_size);
vector<cv::Point2f> OMOGRAFIA(const vector<cv::Vec3f> circles, const Vec4Points vertices, int width, int height);
void Hough_Circles(const cv::Mat& input_img, cv::Mat& img_with_selected_circles, vector<cv::Vec3f>& circles, float min_Dist, float sensibility, int min_Radius, float TH_Circ_A, float TH_Circ_a, float TH_Circ_B, float  th_Ratio_B, float TH_Circ_C, float  th_Ratio_C);
void select_Circles(vector<cv::Vec3f>& circles, float TH_Circ_A, float TH_Circ_a, float TH_Circ_B, float  th_Ratio_B, float TH_Circ_C, float  th_Ratio_C);
void extractFrames(const std::string& videoPath, int frameInterval);
vector <vector<cv::Point2f> > calculate_SquaresVertices(const vector<cv::Vec3f>& circless);
vector<vector<float>> calculate_predictedBoxes(const vector<vector<cv::Point2f>>& allVertices);
void design_Circles(vector<cv::Vec3f>& circles, const cv::Mat& image);
void design_Boxes(const vector < vector<cv::Point2f>>& vertices, cv::Mat& image);


float calculateArea(const vector<float>& box);
float calculateIntersection(const vector<float>& box1, const vector<float>& box2);
float calculateIoU(const vector<float>& box1, const vector<float>& box2);
float calculateAccuracy(const vector<vector<float>>& predictedBoxes, const vector<vector<float>>& groundTruthBoxes, float threshold);


/*--------------------------------------COMMON FUNCTIONS---------------------------------------------*/
void process_general(const function<void(const cv::Mat&, cv::Mat&)>& function, const vector<cv::Mat>& inputImages, vector<cv::Mat>& output_Image);
void process_general(const function<void(const cv::Mat&)>& function, const vector<cv::Mat>& inputImages);
void process_general(const function<void(cv::Mat&)>& function, vector<cv::Mat>& inputImages);

void Loading_images(vector<cv::Mat>& imagess, const string& Folder_Paths);
void Loading_images(vector<cv::Mat>& images, const vector<string>& imagePaths);

void show_image(const char* name_image, const cv::Mat image);
void show_image(const char* name_image, const vector<cv::Mat> Images);
void show_image(const char* name_image, const vector<  vector<cv::Mat> > Images);

void SAVING_image(string folder_path, vector<cv::Mat> salved_images);
void mouse_callback(int event, int x, int y, int flags, void* userdata);

