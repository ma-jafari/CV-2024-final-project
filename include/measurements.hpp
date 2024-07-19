#ifndef Measurements
#define Measurements

using namespace cv;

vector<Rect> compute_bboxes(const vector<Vec3f>& balls);
void loadGroundTruthAndPredictions(const vector<vector<int>>& data, vector<Rect>& boxes);

double computeMeanIoU(const vector<Rect>& gtBoxes, const vector<Rect>& predBoxes);

// Function to compute Average Precision (AP) using Pascal VOC 11-point interpolation
double computeAveragePrecision(const std::vector<Rect>& gtBoxes, const std::vector<Rect>& predBoxes,
    std::vector<int> gtClassIDs, std::vector<int> predClassIDs);

// Function to compute Average Precision (AP) per class
std::vector<double> computeAveragePrecisionPerClass(const std::vector<Rect>& gtBoxes, const std::vector<Rect>& predBoxes,
    std::vector<int> gtClassIDs, std::vector<int> predClassIDs, int numClasses);
#endif // !Measurements
