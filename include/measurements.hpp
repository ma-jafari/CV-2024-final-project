#ifndef Measurements
#define Measurements

using namespace cv;

vector<Rect> compute_bboxes(const vector<Vec3f>& balls);
void loadGroundTruthAndPredictions(const vector<vector<int>>& data, vector<Rect>& boxes);

// Function to compute Average Precision (AP) using Pascal VOC 11-point interpolation
double computeAveragePrecision(const std::vector<Rect>& gtBoxes, const std::vector<Rect>& predBoxes,
    std::vector<int> gtClassIDs, std::vector<int> predClassIDs);

double ComputeIoUPerClass(const Mat& predMask, const Mat& gtMask, int classValue);

#endif // !Measurements
