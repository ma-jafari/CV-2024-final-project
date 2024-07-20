#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <map>

#include "ball_classification.hpp"
#include "ball_detection.hpp"
#include "field_detection.hpp"
#include "measurements.hpp"
#include "functions.h"

using namespace cv;
using namespace std;

int main() {
	string base_path = "../data/";
	vector<string> names = { "game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4",
							 "game2_clip1", "game2_clip2", "game3_clip1", "game3_clip2",
							 "game4_clip1", "game4_clip2" };

	vector<Mat> images;
	vector<Mat> gtMasks;
	vector<vector<Rect>> allBallBboxes;
	vector<Rect> allFieldBboxes;

	// Load images into the vector
	for (const string& name : names) {
		string imagePath = base_path + name + "/frames/frame_first.png";
		Mat image = imread(imagePath);
		if (image.empty()) {
			cerr << "Error loading image file: " << imagePath << endl;
			return -1;
		}
		images.push_back(image);

		string maskPath = base_path + name + "/masks/frame_first.png";
		Mat mask = imread(maskPath, IMREAD_GRAYSCALE);
		if (mask.empty()) {
			cerr << "Error loading mask file: " << maskPath << endl;
			return -1;
		}
		gtMasks.push_back(mask);
	}

	vector<vector<vector<int>>> allLabels; // Vector to store the labels from all label files
	vector<vector<int>> gtClassId(names.size());
	double overallAP = 0.0;
	vector<double> overallAPPerClass(6, 0.0);
	vector<double> classAPSum(6, 0.0); // Accumulate AP for each class
	vector<int> classCount(6, 0); // Count number of instances per class

	// Load images and extract labels from label files
	for (int i = 0; i < names.size(); i++) {
		const string& name = names[i];
		string labelPath = base_path + name + "/bounding_boxes/frame_first_bbox.txt";

		std::vector<std::vector<int>> labels;
		if (extractLabelsFromFile(labelPath, labels)) {
			cout << "For " << name << ":" << endl;
			gtClassId[i].resize(labels.size());
			for (int j = 0; j < labels.size(); j++) {
				const auto& lineLabels = labels[j];
				cout << "  First int: " << lineLabels[0]
					<< ", Second int: " << lineLabels[1]
					<< ", Third int: " << lineLabels[2]
					<< ", Fourth int: " << lineLabels[3]
					<< ", Fifth int: " << lineLabels[4] << endl;
				gtClassId[i][j] = lineLabels[4];
			}
			allLabels.push_back(labels); // Store labels in the vector
		}
		else {
			cerr << "Failed to find all required labels in file: " << labelPath << endl;
		}
	}

	// Output the extracted labels
	for (size_t i = 0; i < allLabels.size(); ++i) {
		cout << "Labels from " << names[i] << ":" << endl;
		for (const auto& lineLabels : allLabels[i]) {
			for (int label : lineLabels) {
				cout << label << " ";
			}
			cout << endl;
		}
	}

	vector<vector<Rect>> predBoxes(names.size());
	vector<vector<int>> predClassIds(names.size());
	vector<double> meanIoUPerClass(6, 0.0);
	vector<int> classIoUCount(6, 0);

	for (int i = 0; i < images.size(); i++) {
		Mat in_img = images[i];
		Mat cutout_table;
		Mat mask = Mat::zeros(in_img.rows, in_img.cols, CV_8UC3);

		// NOTE: We cut find the table boundaries and cut out the table from the rest of the image
		Vec4Points vertices = detect_field(in_img);
		fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
		bitwise_and(in_img, mask, cutout_table);

		Scalar linecolor = Scalar(255, 0, 0);
		int linewidth = LINE_4;
		line(cutout_table, vertices[0], vertices[1], linecolor, linewidth);
		line(cutout_table, vertices[2], vertices[1], linecolor, linewidth);
		line(cutout_table, vertices[2], vertices[3], linecolor, linewidth);
		line(cutout_table, vertices[3], vertices[0], linecolor, linewidth);
		imshow("out", cutout_table);
		Mat cutout_original = cutout_table.clone();
		imshow("mask", mask);

		// NOTE: remove balls on edge of table
		vector<Vec3f> detected_balls = get_balls(cutout_table);
		vector<Vec3f> selected_balls;
		for (int i = 0; i < detected_balls.size(); ++i) {
			Point2f ball = Point2f(detected_balls[i][0], detected_balls[i][1]);
			float radius = detected_balls[i][2];
			if (!(is_ball_near_line(ball, radius, vertices[0], vertices[1]) ||
				is_ball_near_line(ball, radius, vertices[1], vertices[2]) ||
				is_ball_near_line(ball, radius, vertices[2], vertices[3]) ||
				is_ball_near_line(ball, radius, vertices[3], vertices[0]))) {
				selected_balls.push_back(detected_balls[i]);
			}
		}

		// Draw detected balls and bounding boxes
		vector<vector<cv::Point2f>> vertices_boxes = compute_bbox_vertices(selected_balls);
		Mat classifiedImg = in_img.clone();
		vector<int> predClassId(vertices_boxes.size());
		vector<cv::Rect> bbox_rectangles = compute_bboxes(selected_balls);

		// Vectors to store circles for each class
		vector<Point2f> stripped_balls;
		vector<Point2f> solid_balls;
		vector<Point2f> white_balls;
		vector<Point2f> black_balls;

		// Masks for each class
		Mat mask_stripped = Mat::zeros(in_img.size(), CV_8UC1);
		Mat mask_solid = Mat::zeros(in_img.size(), CV_8UC1);
		Mat mask_white = Mat::zeros(in_img.size(), CV_8UC1);
		Mat mask_black = Mat::zeros(in_img.size(), CV_8UC1);

		// Classify balls within the boxes
		for (int j = 0; j < vertices_boxes.size(); j++) {
			vector<Point2f> box = vertices_boxes[j];
			Rect rect(box[0] - Point2f(5, 5),
				box[2] + Point2f(5, 5)); // Assuming box[0] is top-left and box[2] is bottom-right
			Mat roi = in_img(rect);
			Mat ballroi = detectBalls(roi);

			ball_class classifiedBall = classify_ball(ballroi);
			int classId = ReturnBallClass(classifiedBall);
			Point2f center = (box[0] + box[2]) * 0.5;
			float radius = (norm(box[0] - box[2]) * 0.5) * 2 / 3;

			if (classifiedBall == ball_class::STRIPED) {
				rectangle(classifiedImg, rect, Scalar(0, 255, 0), 2);
				stripped_balls.push_back(center);
				circle(mask_stripped, center, radius, Scalar(255), -1);
			}
			else if (classifiedBall == ball_class::SOLID) {
				rectangle(classifiedImg, rect, Scalar(0, 0, 255), 2);
				solid_balls.push_back(center);
				circle(mask_solid, center, radius, Scalar(255), -1);
			}
			else if (classifiedBall == ball_class::CUE) {
				rectangle(classifiedImg, rect, Scalar(255, 255, 255), 2); // White for white ball
				white_balls.push_back(center);
				circle(mask_white, center, radius, Scalar(255), -1);
			}
			else if (classifiedBall == ball_class::EIGHT_BALL) {
				rectangle(classifiedImg, rect, Scalar(0, 0, 0), 2); // Black for black ball
				black_balls.push_back(center);
				circle(mask_black, center, radius, Scalar(255), -1);
			}
			predClassId[j] = classId;
			cout << "Class ID: " << classId << endl; // Print the class ID
		}

		// Create the combined mask
		Mat combinedMask = Mat::zeros(in_img.size(), CV_8UC1);

		// Set field pixels to 5
		Mat fieldMask = Mat::zeros(in_img.size(), CV_8UC1);
		fillPoly(fieldMask, vertices, Scalar(255));
		combinedMask.setTo(Scalar(5), fieldMask);

		// Overlay ball masks with respective values
		Mat ballMasks[4] = { mask_stripped, mask_solid, mask_white, mask_black };
		int values[4] = { 1, 2, 3, 4 };

		for (int k = 0; k < 4; ++k) {
			Mat ballMask = ballMasks[k];
			Mat resizedBallMask;
			resize(ballMask, resizedBallMask, combinedMask.size(), 0, 0, INTER_NEAREST);
			combinedMask.setTo(Scalar(values[k]), resizedBallMask);
		}

		// Compute IoU for each class
		vector<double> classIoUs(6, 0.0);
		for (int c = 0; c < 6; ++c) {
			classIoUs[c] = ComputeIoUPerClass(combinedMask, gtMasks[i], c);
		}

		// Compute mean IoU for this image manually
		double sumIoU = 0.0;
		int numClasses = classIoUs.size();
		for (int c = 0; c < numClasses; ++c) {
			sumIoU += classIoUs[c];
		}
		double meanIoU = (numClasses > 0) ? (sumIoU / numClasses) : 0.0;

		// Print mean IoU results for this image
		cout << "Mean IoU for image " << names[i] << ":" << endl;
		for (int c = 0; c < 6; ++c) {
			cout << "Class " << c << " IoU: " << classIoUs[c] << endl;
		}
		cout << "Mean IoU: " << meanIoU << endl;

		predBoxes[i] = bbox_rectangles;
		predClassIds[i] = predClassId;
	}

	/*// Compute the mean AP for each class
	vector<double> meanAPPerClass(4, 0.0);
	for (int c = 0; c < 4; ++c) {
		if (classCount[c] > 0) {
			meanAPPerClass[c] = classAPSum[c] / classCount[c];
		}
	}

	// Compute the mean of the average precisions (mAP)
	double mAP = 0.0;
	for (int c = 0; c < 4; ++c) {
		mAP += meanAPPerClass[c];
	}
	mAP /= 4.0;

	// Print results
	for (int c = 0; c < 4; ++c) {
		cout << "Class " << c + 1 << " Average Precision: " << meanAPPerClass[c] << endl;
	}
	cout << "Mean Average Precision (mAP): " << mAP << endl;*/

	return 0;
}
