#include <ctime>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "functions.h"
#include "field_detection.hpp"

using namespace cv;
using namespace std;


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
	cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
		3, cv::KMEANS_PP_CENTERS, centers);

	// Convert centers back to 8-bit values and ensure it's of type CV_32F with 3 channels
	centers = centers.reshape(3, centers.rows);

	// Count the number of pixels in each cluster
	std::vector<int> counts(k, 0);
	for (int i = 0; i < labels.rows; ++i) {
		counts[labels.at<int>(i)]++;
	}

	// Find the largest cluster
	int maxIdx = std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));

	// Retrieve the dominant color
	cv::Vec3f dominantColorFloat = centers.at<cv::Vec3f>(maxIdx);

	// Normalize to the range [0, 255]
	cv::Scalar dominantColor(
		dominantColorFloat[0] * 255.0f,
		dominantColorFloat[1] * 255.0f,
		dominantColorFloat[2] * 255.0f
	);

	return dominantColor;
}

// Modified isStriped function
// Modified isStriped function
bool isStriped(const Mat& img) {
	// Convert the image to grayscale
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// Apply adaptive thresholding
	Mat thresholded;
	adaptiveThreshold(gray, thresholded, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

	// Calculate the ratio of white pixels
	double whitePixelRatio = (double)countNonZero(thresholded) / (thresholded.rows * thresholded.cols);

	// You can fine-tune this threshold
	return whitePixelRatio > 0.6;
}

Mat detectBalls(const Mat& src) {
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
		drawContours(result, contours, largestContourIndex, Scalar(0, 255, 0), 2); // Draw the largest contour in green
	}

	// Display the result
	return result;
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

	int size = images.size();
	int temporary_size = 50;
	double total_time = 0.0;

	vector<cv::Mat> masks(size);
	process_general([&](const cv::Mat & in_img, cv::Mat & out_img) {

		// Create the field mask
		cv::Mat field_mask = cv::Mat::zeros(in_img.rows, in_img.cols, CV_8UC3);
		Vec4Points vertices = detect_field(in_img, field_mask);

		// Conversion from Vec4Points to vector<cv::Point>
		vector<cv::Point2i> points(vertices.val, vertices.val + 4);

		// Create a contour for fillPoly
		vector<vector<cv::Point2i>> fillContAll = { points };

		// Create and fill the mask
		cv::Mat mask = cv::Mat::zeros(in_img.rows, in_img.cols, CV_8UC1);
		cv::fillPoly(mask, fillContAll, cv::Scalar(255));

		// Apply the mask to the image
		cv::bitwise_and(in_img, in_img, out_img, mask);

		}, images, masks);
	images = masks;

	int kernel_DILATION = 1;
	int kernel_EROSION = 3;
	float precisione_DIL = 12;
	float precisione_ERO = 11.5;

	float min_Dist = 3;
	int min_Radius = 6;
	float TH_Circ_A = -6;
	float TH_Circ_a = -4;
	float TH_Circ_B = 4;
	float TH_Ratio_B = 0.75;
	float TH_Circ_C = 8;
	float TH_Ratio_C = 0.6;

	clock_t start = clock();

	// DILATION
	vector<cv::Mat> dilated(size);
	process_general([&](const cv::Mat & in_img, cv::Mat & out_img) {

		DILATION(in_img, out_img, kernel_DILATION);

		}, images, dilated);

	// EROSION
	vector<cv::Mat> eroded(size);
	process_general([&](const cv::Mat & in_img, cv::Mat & out_img) {

		EROSION(in_img, out_img, kernel_EROSION);

		}, images, eroded);

	// Calculating CIRCLES_DILATION
	int i = 0;
	vector<cv::Mat> circle_DILATION(size);
	vector< vector<cv::Vec3f> >circles_dilated(size);
	process_general([&](const cv::Mat & in_img, cv::Mat & out_img) {
		vector<cv::Vec3f> cir_dil(temporary_size);
		Hough_Circles(in_img, out_img, cir_dil, min_Dist, precisione_DIL, min_Radius, TH_Circ_A, TH_Circ_a, TH_Circ_B, TH_Ratio_B, TH_Circ_C, TH_Ratio_C);
		circles_dilated[i] = cir_dil;
		i++;
		}, dilated, circle_DILATION);

	// Calculating CIRCLES_EROSION
	i = 0;
	vector<cv::Mat> circle_EROSION(size);
	vector< vector<cv::Vec3f> >circles_erosion(size);
	process_general([&](const cv::Mat & in_img, cv::Mat & out_img) {

		vector<cv::Vec3f> cir_eros(temporary_size);
		Hough_Circles(in_img, out_img, cir_eros, min_Dist, precisione_ERO, min_Radius, TH_Circ_A, TH_Circ_a, TH_Circ_B, TH_Ratio_B, TH_Circ_C, TH_Ratio_C);
		circles_erosion[i] = cir_eros;
		i++;
		}, eroded, circle_EROSION);

	vector< vector<cv::Vec3f> >total_circles(circles_erosion.size());
	for (int i = 0; i < circles_erosion.size(); i++) {
		vector<cv::Vec3f> dil = circles_dilated[i];
		vector<cv::Vec3f> total = circles_erosion[i];

		total.insert(total.end(), dil.begin(), dil.end());

		select_Circles(total, TH_Circ_A, TH_Circ_a, TH_Circ_B, TH_Ratio_B, TH_Circ_C, TH_Ratio_C);
		total_circles[i] = total;
	}

	clock_t end = clock();

	total_time = (float)(end - start) / CLOCKS_PER_SEC * 1000;
	cout << "TOTAL TIME: " << total_time << "ms" << endl;

	// Visualize balls with squared boxes
	for (int i = 0; i < size; i++) {
		vector < vector<cv::Point2f>> vertices_boxes = calculate_SquaresVertices(total_circles[i]);
		//design_Boxes(vertices_boxes, images[i]);

		//Scalar FColor = computeDominantColor(images[i]);
		//cout << "dominante color: " << FColor << endl;

		// Classify balls within the boxes
		for (const auto& box : vertices_boxes) {
			Rect rect(box[0], box[2]); // Assuming box[0] is top-left and box[2] is bottom-right
			Mat roi = images[i](rect);
			Mat ballroi = detectBalls(roi);

			imshow("test", ballroi);
			waitKey(0);
			// Classify the ball using adaptive thresholding
			if (isStriped(ballroi)) {
				rectangle(images[i], rect, Scalar(0, 255, 0), 2); // Green for striped balls
			}
			else {
				rectangle(images[i], rect, Scalar(0, 0, 255), 2); // Red for solid balls
			}
		}

		// Show the final image with rectangles
		imshow("Classified Balls", images[i]);
		waitKey(0);
	}

	cout << "Total processing time: " << total_time << " ms" << endl;

	return 0;
}
