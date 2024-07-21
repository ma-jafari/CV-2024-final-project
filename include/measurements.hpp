// Ali Jafari
#ifndef Measurements
#define Measurements
#include "ball_classification.hpp"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

vector<Rect> compute_bboxes(const vector<Vec3f> &balls);
void loadGroundTruthAndPredictions(const vector<vector<int>> &data,
                                   vector<Rect> &boxes);

// Function to compute Average Precision (AP) using Pascal VOC 11-point interpolation
double computeAveragePrecision(const std::vector<Rect>& gtBoxes, const std::vector<Rect>& predBoxes,
    std::vector<int> gtClassIDs, std::vector<int> predClassIDs);

double ComputeIoUPerClass(const Mat& predMask, const Mat& gtMask, int classValue);

void ComputeMeanIoU(Mat frame, Mat gtMask, Vec4Points vertices, string path, vector<Mat> ballMasks);

// Function to compute Average Precision (AP) using Pascal VOC 11-point
// interpolation
double computeAP(const std::vector<Rect> &gtBoxes,
                 const std::vector<Rect> &predBoxes,
                 std::vector<ball_class> &gtClassIDs,
                 std::vector<ball_class> &predClassIDs, ball_class classID);

double computeMeanAP(const vector<Rect> &gtBoxes, const vector<Rect> &predBoxes,
                     vector<ball_class> &gtClassIDs,
                     vector<ball_class> &predClassIDs);

bool extractLabelsFromFile(const std::string &filename,
                           std::vector<std::vector<int>> &allLabels);
#endif // !Measurements
