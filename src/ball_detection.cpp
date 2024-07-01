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

#include "functions.h"

using namespace cv;
using namespace std;

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
