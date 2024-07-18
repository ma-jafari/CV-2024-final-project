#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "ball_classification.hpp"

using namespace std;
using namespace cv;

// Function to compute bounding boxes from detected balls
vector<Rect> compute_bboxes(const vector<Vec3f>& balls) {
	vector<Rect> bboxes;
	for (const auto& ball : balls) {
		int x = static_cast<int>(ball[0] - ball[2]);
		int y = static_cast<int>(ball[1] - ball[2]);
		int width = static_cast<int>(2 * ball[2]);
		int height = static_cast<int>(2 * ball[2]);
		bboxes.emplace_back(x, y, width, height);
	}
	return bboxes;
}

// Function to load ground truth and prediction data
void loadGroundTruthAndPredictions(const vector<vector<int>> & data, vector<Rect> & boxes) {
	for (const auto& boxData : data) {
		int x = boxData[0];
		int y = boxData[1];
		int width = boxData[2];
		int height = boxData[3];
		boxes.emplace_back(x, y, width, height);
	}
}

// Function to compute Intersection over Union (IoU)
double computeIoU(const Rect & gtBox, const Rect & predBox) {
	int x1 = max(gtBox.x, predBox.x);
	int y1 = max(gtBox.y, predBox.y);
	int x2 = min(gtBox.x + gtBox.width, predBox.x + predBox.width);
	int y2 = min(gtBox.y + gtBox.height, predBox.y + predBox.height);

	int intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
	int gtBoxArea = gtBox.width * gtBox.height;
	int predBoxArea = predBox.width * predBox.height;
	int unionArea = gtBoxArea + predBoxArea - intersectionArea;

	return static_cast<double>(intersectionArea) / unionArea;
}

// Function to compute mean Intersection over Union (mIoU) for a single image
double computeMeanIoU(const vector<Rect> & gtBoxes, const vector<Rect> & predBoxes) {
	double sumIoU = 0.0;
	int count = 0;
	for (const auto& gtBox : gtBoxes) {
		double maxIoU = 0.0;
		for (const auto& predBox : predBoxes) {
			double iou = computeIoU(gtBox, predBox);
			if (iou > maxIoU) {
				maxIoU = iou;
			}
		}
		sumIoU += maxIoU;
		++count;
	}
	return count > 0 ? sumIoU / count : 0.0;
}

// Function to compute Precision and Recall
pair<double, double> computePrecisionRecall(const vector<Rect> & gtBoxes, const vector<Rect> & predBoxes,
	double iouThreshold, vector<int> gtClassIDs, vector<int> predClassIDs) {
	int tp = 0; // True positives
	int fp = 0; // False positives
	int fn = 0; // False negatives

	for (size_t i = 0; i < predBoxes.size(); i++) {
		bool matched = false;
		for (size_t j = 0; j < gtBoxes.size(); j++) {
			// Only consider boxes with the same class ID
			if (predClassIDs[i] == gtClassIDs[j]) {
				double iou = computeIoU(gtBoxes[j], predBoxes[i]);
				if (iou >= iouThreshold) {
					++tp;
					matched = true;
					break; // Only match each ground truth box once
				}
			}
		}
		if (!matched) {
			++fp;
		}
	}

	for (size_t j = 0; j < gtBoxes.size(); j++) {
		bool matched = false;
		for (size_t i = 0; i < predBoxes.size(); i++) {
			if (predClassIDs[i] == gtClassIDs[j]) {
				double iou = computeIoU(gtBoxes[j], predBoxes[i]);
				if (iou >= iouThreshold) {
					matched = true;
					break; // Only match each ground truth box once
				}
			}
		}
		if (!matched) {
			++fn;
		}
	}

	double precision = (tp + fp) > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
	double recall = (tp + fn) > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;

	return make_pair(precision, recall);
}	

// Function to compute Average Precision (AP) using Pascal VOC 11-point interpolation
double computeAveragePrecision(const vector<Rect>& gtBoxes, const vector<Rect>& predBoxes,
	vector<int> gtClassIDs, vector<int> predClassIDs) {
	double ap = 0.0;
	std::vector<double> precision;
	std::vector<double> recall;
	std::vector<double> pascalPts = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };

	// Compute precision-recall for each threshold
	for (auto& t : pascalPts) {
		auto [prec, rec] = computePrecisionRecall(gtBoxes, predBoxes, t, gtClassIDs, predClassIDs);
		precision.push_back(prec);
		recall.push_back(rec);
	}

	// Compute AP using 11-point interpolation
	for (size_t i = 0; i < pascalPts.size(); i++) {
		double max_precision = 0;
		for (size_t j = 0; j < recall.size(); j++) {
			if (recall[j] >= pascalPts[i] && precision[j] > max_precision) {
				max_precision = precision[j];
			}
		}
		ap += max_precision;
	}

	return ap / pascalPts.size(); 
}

// Function to compute Average Precision (AP) using Pascal VOC 11-point interpolation for each class
vector<double> computeAveragePrecisionPerClass(const vector<Rect>& gtBoxes, const vector<Rect>& predBoxes,
                                               vector<int> gtClassIDs, vector<int> predClassIDs, int numClasses) {
    vector<double> ap_per_class(numClasses, 0.0);
    vector<vector<Rect>> gtBoxesPerClass(numClasses);
    vector<vector<Rect>> predBoxesPerClass(numClasses);

    // Separate ground truth and predicted boxes by class ID
    for (size_t i = 0; i < gtBoxes.size(); ++i) {
        int classId = gtClassIDs[i] - 1;
        gtBoxesPerClass[classId].push_back(gtBoxes[i]);
    }

    for (size_t i = 0; i < predBoxes.size(); ++i) {
        int classId = predClassIDs[i] - 1;
        predBoxesPerClass[classId].push_back(predBoxes[i]);
    }

    // Compute AP for each class
    for (int c = 0; c < numClasses; ++c) {
        ap_per_class[c] = computeAveragePrecision(gtBoxesPerClass[c], predBoxesPerClass[c], gtClassIDs, predClassIDs);
    }

    return ap_per_class;
}

