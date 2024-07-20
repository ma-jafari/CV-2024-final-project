#ifndef Measurements
#define Measurements
#include "ball_classification.hpp"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

vector<Rect> compute_bboxes(const vector<Vec3f> &balls);
void loadGroundTruthAndPredictions(const vector<vector<int>> &data,
                                   vector<Rect> &boxes);

double computeMeanIoU(const vector<Rect> &gtBoxes,
                      const vector<Rect> &predBoxes);

// Function to compute Average Precision (AP) using Pascal VOC 11-point
// interpolation
double computeAP(const std::vector<Rect> &gtBoxes,
                 const std::vector<Rect> &predBoxes,
                 std::vector<ball_class> &gtClassIDs,
                 std::vector<ball_class> &predClassIDs, ball_class classID);

// Function to compute Average Precision (AP) per class
std::vector<double> computeAveragePrecisionPerClass(
    const std::vector<Rect> &gtBoxes, const std::vector<Rect> &predBoxes,
    std::vector<int> gtClassIDs, std::vector<int> predClassIDs, int numClasses);
#endif // !Measurements
