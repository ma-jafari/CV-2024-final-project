#ifndef Measurements
#define Measurements

using namespace cv;

vector<Rect> compute_bboxes(const vector<Vec3f>& balls);
void loadGroundTruthAndPredictions(const vector<vector<int>>& data, vector<Rect>& boxes);

double computeIoU(const Rect& gtBox, const Rect& predBox);
double computeMeanIoU(const vector<Rect>& gtBoxes, const vector<Rect>& predBoxes);

pair<double, double> computePrecisionRecall(const vector<Rect>& gtBoxes, const vector<Rect>& predBoxes, double iouThreshold);
double computeAveragePrecision(const vector<Rect>& gtBoxes, const vector<Rect>& predBoxes);

#endif // !Measurements
