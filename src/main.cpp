#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
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

#include "field_detection.hpp"
#include "functions.h"

using namespace cv;
using namespace std;

bool extractLabelsFromFile(const std::string& filename, std::vector<std::vector<int>>& allLabels) {
	std::ifstream inputFile(filename);  // Open the file for reading
	if (!inputFile.is_open()) {         // Check if the file opened successfully
		std::cerr << "Failed to open file" << std::endl;
		return false;
	}

	std::string line;
	bool foundLabel = false;

	// Read the subsequent lines
	while (std::getline(inputFile, line)) {
		std::istringstream iss(line);
		int label1, label2, label3, label4, label5;

		if (iss >> label1 >> label2 >> label3 >> label4 >> label5) {
			allLabels.push_back({ label1, label2, label3, label4, label5 });
			foundLabel = true;
		}
	}

	inputFile.close();  // Close the file

	return foundLabel;
}

cv::Scalar computeDominantColor(const cv::Mat& img) {
	int k = 3;
	// Check if the image type is CV_8UC3
	if (img.type() != CV_8UC3) {
		throw std::runtime_error("The image is not of type CV_8UC3!");
	}

	// Convert the image to a float type for k-means
	cv::Mat data;
	img.convertTo(data, CV_32F);

	// Reshape the image to a 2D array of pixels
	data = data.reshape(1, data.total());

	// Define criteria and apply k-means clustering
	cv::Mat labels, centers;
	cv::kmeans(data, k, labels,
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
			10, 1.0),
		3, cv::KMEANS_PP_CENTERS, centers);

	// Convert centers back to 8-bit values and ensure it's of type CV_32F with 3
	// channels
	centers = centers.reshape(3, centers.rows);

	// Count the number of pixels in each cluster
	std::vector<int> counts(k, 0);
	for (int i = 0; i < labels.rows; ++i) {
		counts[labels.at<int>(i)]++;
	}

	// Find the largest cluster
	int maxIdx = std::distance(counts.begin(),
		std::max_element(counts.begin(), counts.end()));

	// Retrieve the dominant color
	cv::Vec3f dominantColorFloat = centers.at<cv::Vec3f>(maxIdx);

	// Normalize to the range [0, 255]
	cv::Scalar dominantColor(dominantColorFloat[0] * 255.0f,
		dominantColorFloat[1] * 255.0f,
		dominantColorFloat[2] * 255.0f);

	return dominantColor;
}

// Modified isStriped function
bool isStriped(const Mat& img) {
	// Convert the image to grayscale
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// Apply adaptive thresholding
	Mat thresholded;

	namedWindow("gray");
	resizeWindow("gray", 400, 300);
	// equalizeHist(gray, gray);
	imshow("gray", gray);
	// threshold(gray, thresholded, 210, 255, THRESH_BINARY);
	/*threshold(gray, thresholded, 210, 255,
			  THRESH_BINARY); // FIX: for now the threshold is good, we need to
							  // try with hist eq
	*/
	constexpr int lt = 120;
	inRange(img, Scalar(lt, lt, lt), Scalar(255, 255, 255), thresholded);
	// FIX: togli dilation
	namedWindow("thresholded");
	resizeWindow("thresholded", 400, 300);
	imshow("thresholded", thresholded);
	dilate(thresholded, thresholded,
		getStructuringElement(MORPH_RECT, Size(3, 3)));
	// Calculate the ratio of white pixels
	double whitePixelRatio =
		(double)countNonZero(thresholded) / (thresholded.rows * thresholded.cols);

	cout << (whitePixelRatio > 0.05 ? "striped" : "solid") << endl;

	waitKey();
	// You can fine-tune this threshold
	return whitePixelRatio > 0.05;
}

Mat detectBalls(const Mat & src) {
	// Convert to grayscale
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	// Apply GaussianBlur to reduce noise and improve contour detection
	Mat blurred;
	GaussianBlur(gray, blurred, Size(5, 5), 0);

	// Apply thresholding
	Mat binary;
	threshold(blurred, binary, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);

	// Find contours
	vector<vector<Point>> contours;
	findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Find the largest contour
	int largestContourIndex = -1;
	double largestArea = 0;
	for (size_t i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > largestArea) {
			largestArea = area;
			largestContourIndex = i;
		}
	}

	// Draw the largest contour
	Mat result = src.clone();
	if (largestContourIndex != -1) {
		// drawContours(result, contours, largestContourIndex, Scalar(0, 255, 0),
		//             2); // Draw the largest contour in green
	}

	// Display the result
	return result;
}

std::vector<Vec3f> get_balls(cv::Mat & in_img) {

	int kernel_DILATION = 3;
	int kernel_EROSION = 3;
	float precisione_DIL = 13; // 12;
	float precisione_ERO = 11.5;


	ball_detection_params ball_params;
	ball_params.min_Dist = 2;
	ball_params.min_Radius = 8;
	ball_params.TH_Circ_A = -6;
	ball_params.TH_Circ_a = -4;
	ball_params.TH_Circ_B = 4;
	ball_params.TH_Ratio_B = 0.75;
	ball_params.TH_Circ_C = 8;
	ball_params.TH_Ratio_C = 0.6;

	Mat dilated;
	Mat eroded;
	erode_image(in_img, dilated, kernel_DILATION);

	erode_image(in_img, eroded, kernel_EROSION);
	/*
	imshow("dilated", dilated);
	imshow("eroded", eroded);
	*/

	Mat dilated_canny, eroded_canny;
	Canny(dilated, dilated_canny, 300, 300);
	Canny(eroded, eroded_canny, 300, 300);
	// imshow("dilcanny", dilated_canny);
	// imshow("eroded canny", eroded_canny);
	dilated = dilated_canny;
	eroded = eroded_canny;

	// dilated circle detection
	vector<cv::Vec3f> circles_dilated;
	get_circles(dilated, circles_dilated, precisione_DIL, ball_params);

	// eroded circle detection
	cv::Mat circle_EROSION;
	vector<cv::Vec3f> circles_erosion;
	get_circles(eroded, circles_erosion, precisione_ERO, ball_params);

	vector<cv::Vec3f> dil = circles_dilated;
	vector<cv::Vec3f> total = circles_erosion;

	total.insert(total.end(), dil.begin(), dil.end());

	select_circles(total, ball_params);
	cout << total.size() << endl;

	return total;
}

bool is_ball_near_line(Point2f ball_pos, float radius, Point2f pointA,
	Point2f pointB) {
	float temp = (pointB.y - pointA.y) * ball_pos.x -
		(pointB.x - pointA.x) * ball_pos.y + pointB.x * pointA.y -
		pointB.y * pointA.x;
	float distance = fabs(temp) / norm(pointB - pointA);
	return distance < 1.0f * radius;
}

int main() {
	string base_path = "../data/";
	string names[] = { "game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4",
					  "game2_clip1", "game2_clip2", "game3_clip1", "game3_clip2",
					  "game4_clip1", "game4_clip2" };

	vector<Mat> images;

	// Load images into the vector
	for (const string& name : names) {
		string imagePath = base_path + name + "/frames/frame_first.png";
		Mat image = imread(imagePath);
		if (image.empty()) {
			cerr << "Error loading image file: " << imagePath << endl;
			return -1;
		}
		images.push_back(image);
	}

	vector<vector<vector<int>>> allLabels; // Vector to store the labels from all label files

	// Load images and extract labels from label files
	for (const string& name : names) {
		string labelPath = base_path + name + "/bounding_boxes/frame_first_bbox.txt";

		std::vector<std::vector<int>> labels;
		if (extractLabelsFromFile(labelPath, labels)) {
			cout << "For " << name << ":" << endl;
			for (const auto& lineLabels : labels) {
				cout << "  First int: " << lineLabels[0] << ", Second int: " << lineLabels[1] << ", Third int: " << lineLabels[2]
					<< ", Fourth int: " << lineLabels[3] << ", Fifth int: " << lineLabels[4] << endl;
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

	for (auto& in_img : images) {
		Mat cutout_table;
		Mat mask = Mat::zeros(in_img.rows, in_img.cols, CV_8UC3);

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

		// NOTE: SHOW BALLS DETECTED
		vector<vector<cv::Point2f>> vertices_boxes =
			compute_bbox_vertices(selected_balls);
		draw_bboxes(vertices_boxes, in_img);
		circle(cutout_table, vertices[0], 20, Scalar(0, 0, 255));
		circle(cutout_table, vertices[1], 20, Scalar(0, 0, 255));
		circle(cutout_table, vertices[2], 20, Scalar(0, 0, 255));
		circle(cutout_table, vertices[3], 20, Scalar(0, 0, 255));
		imshow("vertices", cutout_table);

		//

		// Scalar FColor = computeDominantColor(images[i]);
		// cout << "dominante color: " << FColor << endl;

		// Classify balls within the boxes
		for (const auto& box : vertices_boxes) {
			Rect rect(box[0] - Point2f(5, 5),
				box[2] + Point2f(5, 5)); // Assuming box[0] is top-left and
										 // box[2] is bottom-right
			Mat roi = in_img(rect);
			Mat ballroi = detectBalls(roi);

			namedWindow("test");

			resizeWindow("test", 400, 300);
			imshow("test", ballroi);
			// Classify the ball using adaptive thresholding
			if (isStriped(ballroi)) {
				rectangle(in_img, rect, Scalar(0, 255, 0),
					2); // Green for striped balls
			}
			else {
				rectangle(in_img, rect, Scalar(0, 0, 255), 2); // Red for solid balls
			}
		}
		/*
		// Visualize balls with squared boxes

		  // Show the final image with rectangles
		  // imshow("Classified Balls", images[i]);
		  waitKey(0);
		}
		*/
	}

	return 0;
}
