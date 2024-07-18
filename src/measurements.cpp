
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
void loadGroundTruthAndPredictions(const vector<vector<int>>& data, vector<Rect>& boxes) {
	for (const auto& boxData : data) {
		int x = boxData[0];
		int y = boxData[1];
		int width = boxData[2];
		int height = boxData[3];
		boxes.emplace_back(x, y, width, height);
	}
}

// Function to compute Intersection over Union (IoU)
double computeIoU(const Rect& gtBox, const Rect& predBox) {
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
double computeMeanIoU(const vector<Rect>& gtBoxes, const vector<Rect>& predBoxes) {
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
pair<double, double> computePrecisionRecall(const vector<Rect>& gtBoxes, const vector<Rect>& predBoxes, double iouThreshold) {
	int tp = 0; // True positives
	int fp = 0; // False positives
	int fn = 0; // False negatives

	vector<bool> gtMatched(gtBoxes.size(), false);

	for (const auto& predBox : predBoxes) {
		bool matched = false;
		for (size_t i = 0; i < gtBoxes.size(); ++i) {
			if (computeIoU(gtBoxes[i], predBox) >= iouThreshold) {
				if (!gtMatched[i]) {
					gtMatched[i] = true;
					matched = true;
					++tp;
					break;
				}
			}
		}
		if (!matched) {
			++fp;
		}
	}

	for (bool matched : gtMatched) {
		if (!matched) {
			++fn;
		}
	}

	double precision = tp + fp > 0 ? static_cast<double>(tp) / (static_cast<double>(tp)
		+ static_cast<double>(fp)) : 0.0;
	double recall = tp + fn > 0 ? static_cast<double>(tp) / (static_cast<double>(tp)
		+ static_cast<double>(fn)) : 0.0;

	return make_pair(precision, recall);
}

// Function to compute Average Precision (AP)
double computeAveragePrecision(const vector<Rect> & gtBoxes, const vector<Rect> & predBoxes) {
	vector<double> precisions;
	vector<double> recalls;
	vector<double> iouThresholds = { 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95 };

	for (double iouThreshold : iouThresholds) {
		auto [precision, recall] = computePrecisionRecall(gtBoxes, predBoxes, iouThreshold);
		precisions.push_back(precision);
		recalls.push_back(recall);
	}

	double ap = 0.0;
	for (size_t i = 0; i < precisions.size(); ++i) {
		ap += precisions[i];
	}

	return ap / precisions.size();
}