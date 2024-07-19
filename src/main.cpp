﻿#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
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

#include "ball_classification.hpp"
#include "ball_detection.hpp"
#include "ball_tracking.hpp"
#include "field_detection.hpp"
#include "functions.h"
#include "measurements.hpp"

using namespace cv;
using namespace std;

int main() {
<<<<<<< HEAD
  string base_path = "../data/";
  string names[] = {"game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4",
                    "game2_clip1", "game2_clip2", "game3_clip1", "game3_clip2",
                    "game4_clip1", "game4_clip2"};
=======
	string base_path = "../data/";
	vector<string> names = { "game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4",
					  "game2_clip1", "game2_clip2", "game3_clip1", "game3_clip2",
					  "game4_clip1", "game4_clip2" };
>>>>>>> measurements

  vector<Mat> images;

  // Load images into the vector
  for (const string &name : names) {
    string imagePath = base_path + name + "/frames/frame_first.png";
    Mat image = imread(imagePath);
    if (image.empty()) {
      cerr << "Error loading image file: " << imagePath << endl;
      return -1;
    }
    images.push_back(image);
  }

<<<<<<< HEAD
  vector<vector<vector<int>>>
      allLabels; // Vector to store the labels from all label files

  // Load images and extract labels from label files
  for (const string &name : names) {
    string labelPath =
        base_path + name + "/bounding_boxes/frame_first_bbox.txt";

    std::vector<std::vector<int>> labels;
    if (extractLabelsFromFile(labelPath, labels)) {
      cout << "For " << name << ":" << endl;
      for (const auto &lineLabels : labels) {
        cout << "  First int: " << lineLabels[0]
             << ", Second int: " << lineLabels[1]
             << ", Third int: " << lineLabels[2]
             << ", Fourth int: " << lineLabels[3]
             << ", Fifth int: " << lineLabels[4] << endl;
      }
      allLabels.push_back(labels); // Store labels in the vector
    } else {
      cerr << "Failed to find all required labels in file: " << labelPath
           << endl;
    }
  }

  // Output the extracted labels
  for (size_t i = 0; i < allLabels.size(); ++i) {
    cout << "Labels from " << names[i] << ":" << endl;
    for (const auto &lineLabels : allLabels[i]) {
      for (int label : lineLabels) {
        cout << label << " ";
      }
      cout << endl;
    }
  }

  for (int i = 0; i < images.size(); i++) {
    Mat in_img = images[i];
    Mat cutout_table;
    Mat mask = Mat::zeros(in_img.rows, in_img.cols, CV_8UC3);
=======
	vector<vector<vector<int>>>
		allLabels; // Vector to store the labels from all label files
	vector<vector<int>> gtClassId(names.size());
	double overallAP = 0.0;
	vector<double> overallAPPerClass(4, 0.0);
	vector<double> classAPSum(4, 0.0); // Accumulate AP for each class
	vector<int> classCount(4, 0); // Count number of instances per class

	// Load images and extract labels from label files
	for (int i = 0; i < names.size(); i++) {
		const string& name = names[i];
		string labelPath =
			base_path + name + "/bounding_boxes/frame_first_bbox.txt";

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
			cerr << "Failed to find all required labels in file: " << labelPath
				<< endl;
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
	
	for (int i = 0; i < images.size(); i++) {
		Mat in_img = images[i];
		Mat cutout_table;
		Mat mask = Mat::zeros(in_img.rows, in_img.cols, CV_8UC3);
>>>>>>> measurements

    // NOTE: We cut find the table boundaries and cut out the table
    // from the rest of the image
    Vec4Points vertices = detect_field(in_img);
    fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    bitwise_and(in_img, mask, cutout_table);

    Scalar linecolor = Scalar(255, 0, 0);
    int linewidth = LINE_4;
    line(cutout_table, vertices[0], vertices[1], linecolor, linewidth);
    line(cutout_table, vertices[2], vertices[1], linecolor, linewidth);
    line(cutout_table, vertices[2], vertices[3], linecolor, linewidth);
    line(cutout_table, vertices[3], vertices[0], linecolor, linewidth);
    // imshow("out", cutout_table);
    Mat cutout_original = cutout_table.clone();
    // imshow("mask", mask);

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

<<<<<<< HEAD
    // NOTE: SHOW BALLS DETECTED
    vector<vector<cv::Point2f>> vertices_boxes =
        compute_bbox_vertices(selected_balls);
    // draw_bboxes(vertices_boxes, in_img);
    circle(cutout_table, vertices[0], 20, Scalar(0, 0, 255));
    circle(cutout_table, vertices[1], 20, Scalar(0, 0, 255));
    circle(cutout_table, vertices[2], 20, Scalar(0, 0, 255));
    circle(cutout_table, vertices[3], 20, Scalar(0, 0, 255));
    // imshow("vertices", cutout_table);
=======
		// draw_bboxes(vertices_boxes, in_img);
		circle(cutout_table, vertices[0], 20, Scalar(0, 0, 255));
		circle(cutout_table, vertices[1], 20, Scalar(0, 0, 255));
		circle(cutout_table, vertices[2], 20, Scalar(0, 0, 255));
		circle(cutout_table, vertices[3], 20, Scalar(0, 0, 255));
		imshow("vertices", cutout_table);
>>>>>>> measurements

    //

<<<<<<< HEAD
    // Scalar FColor = computeDominantColor(images[i]);
    // cout << "dominante color: " << FColor << endl;

    // For demonstration
    Mat classifiedImg = in_img.clone();
=======
		// Scalar FColor = computeDominantColor(images[i]);
		// cout << "dominante color: " << FColor << endl;
		
		// Draw detected balls and bounding boxes
		vector<vector<cv::Point2f>> vertices_boxes = compute_bbox_vertices(selected_balls);
		Mat classifiedImg = in_img.clone();
		vector<cv::Rect> bbox_rectangles = compute_bboxes(selected_balls);
		vector<int> predClassId(vertices_boxes.size());

		// vector<cv::Rect> bbox_rectangles;
		// Classify balls within the boxes
		for (int j = 0; j < vertices_boxes.size(); j++) {
			vector<Point2f> box = vertices_boxes[j];
			Rect rect(box[0] - Point2f(5, 5),
				box[2] + Point2f(5, 5)); // Assuming box[0] is top-left and
										 // box[2] is bottom-right
			// bbox_rectangles.push_back(rect);
			Mat roi = in_img(rect);
			Mat ballroi = detectBalls(roi);
>>>>>>> measurements

    // vector<cv::Rect> bbox_rectangles;
    // Classify balls within the boxes
    for (const auto &box : vertices_boxes) {
      Rect rect(box[0] - Point2f(5, 5),
                box[2] + Point2f(5, 5)); // Assuming box[0] is top-left and
                                         // box[2] is bottom-right
      // bbox_rectangles.push_back(rect);
      Mat roi = in_img(rect);
      Mat ballroi = detectBalls(roi);

<<<<<<< HEAD
      // namedWindow("test");

      // resizeWindow("test", 400, 300);
      // imshow("test", ballroi);
      // Classify the ball using adaptive thresholding
      if (classify_ball(ballroi) == ball_class::STRIPED) {
        rectangle(classifiedImg, rect, Scalar(0, 255, 0),
                  2); // Green for striped balls
      } else {
        rectangle(classifiedImg, rect, Scalar(0, 0, 255),
                  2); // Red for solid balls
      }
    }
    imshow("classified image", classifiedImg);
    // waitKey(0);

    vector<cv::Rect> bbox_rectangles = compute_bboxes(selected_balls);
=======
			// resizeWindow("test", 400, 300);
			// imshow("test", ballroi);
			// Classify the ball using adaptive thresholding
			ball_class classifiedBall = classify_ball(ballroi);
			int classId = ReturnBallClass(classifiedBall);
			if (classifiedBall == ball_class::STRIPED) {
				rectangle(classifiedImg, rect, Scalar(0, 255, 0),
					2); // Green for striped balls
			}
			else {
				rectangle(classifiedImg, rect, Scalar(0, 0, 255), 2); // Red for solid balls
			}
			predClassId[j] = classId;
		}
		imshow("classified image", classifiedImg);
		waitKey(0);

		vector<Rect> gtBoxes; // Ground truth bounding boxes for this image
		vector<Rect> predBoxes; // Predicted bounding boxes for this image
>>>>>>> measurements

    vector<Rect> gtBoxes;   // Ground truth bounding boxes for this image
    vector<Rect> predBoxes; // Predicted bounding boxes for this image

<<<<<<< HEAD
    // Load ground truth and predictions
    loadGroundTruthAndPredictions(allLabels[i], gtBoxes);
    predBoxes = bbox_rectangles;

    // FIX: for now I use these bounding boxes for the tracking

    track_balls(names[i], predBoxes);
    /// END OF TRACKING
=======
		double meanIoU = computeMeanIoU(gtBoxes, predBoxes);
		double averagePrecision = computeAveragePrecision(gtBoxes, predBoxes, gtClassId[i], predClassId);
		vector<double> ap_per_class = computeAveragePrecisionPerClass(gtBoxes, predBoxes, gtClassId[i], predClassId, 4);

		// Print results for each image
		cout << "Image " << names[i] << ":" << endl;
		cout << "Mean IoU: " << meanIoU << endl;
		cout << "Average Precision: " << averagePrecision << endl;

		for (int c = 0; c < 4; ++c) {
			cout << "Class " << c + 1 << " AP: " << ap_per_class[c] << endl;
			overallAPPerClass[c] += ap_per_class[c];
		}

		// Calculate and print mAP for the current image
		double mAP = 0.0;
		for (int c = 0; c < 4; ++c) {
			mAP += ap_per_class[c];
		}
		mAP /= 4.0;

		cout << "Mean Average Precision (mAP) for image " << names[i] << ": " << mAP << endl;

		// Accumulate AP for overall mAP calculation
		overallAP += averagePrecision;

		// Visualize ground truth and predicted boxes (visualization code omitted for brevity)
		Mat Bboxes_img = in_img.clone();
>>>>>>> measurements

    // Compute evaluation metrics
    double meanIoU = computeMeanIoU(gtBoxes, predBoxes);
    double averagePrecision = computeAveragePrecision(gtBoxes, predBoxes);

    // Print results
    cout << "Image " << names[i] << ":" << endl;
    cout << "Mean IoU: " << meanIoU << endl;
    cout << "Average Precision: " << averagePrecision << endl;
    cout << endl;

<<<<<<< HEAD
    Mat Bboxes_img = in_img.clone();

    // Visualize ground truth and predicted boxes
    for (const auto &gtBox : gtBoxes) {
      rectangle(Bboxes_img, gtBox, Scalar(0, 0, 255),
                2); // Red rectangles for ground truth boxes
    }
    for (const auto &predBox : predBoxes) {
      rectangle(Bboxes_img, predBox, Scalar(255, 0, 0),
                2); // Blue rectangles for predicted boxes
    }

    imshow("Bounding Boxes", Bboxes_img);
    waitKey(0);
  }

  return 0;
}
=======
	// Compute the mean AP for each class
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
	cout << "Mean Average Precision (mAP): " << mAP << endl;

	return 0;
}
>>>>>>> measurements
