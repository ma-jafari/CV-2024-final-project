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
vector<vector<float>> paramValues = {
		{1.0f, 2.0f},         // Possibili valori per kernel_DILATION
		{2.0f, 3.0f, 4.0f},         // Possibili valori per kernel_EROSION
		{11.5f, 12.0f, 12.5f},      // Possibili valori per precisione_DIL
		{10.0f, 10.5f, 11.0f},       // Possibili valori per precisione_ERO
		{2.0f, 3.0f, 4.0f},         // Possibili valori per min_Dist
		{5.0f, 6.0f, 7.0f},         // Possibili valori per min_Radius
		{-4.0f, -3.0f, -2.0f},      // Possibili valori per TH_Circ_A
		{1.0f, 2.0f, 3.0f},         // Possibili valori per TH_Circ_B
		{0.8f, 0.85f, 0.9f}          // Possibili valori per TH_Ratio
};

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

	  // Creazione della maschera del campo
	  cv::Mat field_mask = cv::Mat::zeros(in_img.rows, in_img.cols, CV_8UC3);
	  Vec4Points vertices = detect_field(in_img, field_mask);

	  // Temporary vars for visualization
	  Vec4Points visualVertices = vertices;
	  cv::Mat tempIn = in_img.clone();

	  circle(tempIn, visualVertices[0], 20, Scalar(255, 100, 255), LINE_8);
	  circle(tempIn, visualVertices[1], 20, Scalar(255, 100, 255), LINE_8);
	  circle(tempIn, visualVertices[2], 20, Scalar(255, 100, 255), LINE_8);
	  circle(tempIn, visualVertices[3], 20, Scalar(255, 100, 255), LINE_8);
	  imshow("Vertices", tempIn);
	  imshow("Mask", field_mask);
	  waitKey(0);

	  // Conversione di Vec4Points in vector<cv::Point>
	  vector<cv::Point2i> points(vertices.val, vertices.val + 4);

	  // Creazione di un contorno per fillPoly
	  vector<vector<cv::Point2i>> fillContAll = { points };

	  // Creazione e riempimento della maschera
	  cv::Mat mask = cv::Mat::zeros(in_img.rows, in_img.cols, CV_8UC1);
	  cv::fillPoly(mask, fillContAll, cv::Scalar(255));

	  // Applicazione della maschera all'immagine
	  cv::bitwise_and(in_img, in_img, out_img, mask);

	  /*cv::imshow("Immagine Originale", in_img);
	  cv::imshow("Immagine Mascherata", masked_image);
	  cv::waitKey(0);*/

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

  //DILATION
  vector<cv::Mat> dilated(size);
  process_general([&](const cv::Mat & in_img, cv::Mat & out_img) {

	  DILATION(in_img, out_img, kernel_DILATION); // (input_image, output_image, dilation_size)

	  }, images, dilated);

  //EROSIONN
  vector<cv::Mat> eroded(size);
  process_general([&](const cv::Mat & in_img, cv::Mat & out_img) {

	  EROSION(in_img, out_img, kernel_EROSION);

	  }, images, eroded);

  //calculating CIRCLES_DILATION
  int i = 0;
  vector<cv::Mat> circle_DILATION(size);
  vector< vector<cv::Vec3f> >circles_dilated(size);
  process_general([&](const cv::Mat & in_img, cv::Mat & out_img) {
	  vector<cv::Vec3f> cir_dil(temporary_size);
	  Hough_Circles(in_img, out_img, cir_dil, min_Dist, precisione_DIL, min_Radius, TH_Circ_A, TH_Circ_a, TH_Circ_B, TH_Ratio_B, TH_Circ_C, TH_Ratio_C); //parametro per calcolare pi� o meno ccerhi // 10 <parametro< 20
	  circles_dilated[i] = cir_dil;
	  i++;
	  }, dilated, circle_DILATION);

  //calculating CIRCLES_EROSION
  i = 0;
  vector<cv::Mat> circle_EROSION(size);
  vector< vector<cv::Vec3f> >circles_erosion(size);
  process_general([&](const cv::Mat & in_img, cv::Mat & out_img) {

	  vector<cv::Vec3f> cir_eros(temporary_size);
	  Hough_Circles(in_img, out_img, cir_eros, min_Dist, precisione_ERO, min_Radius, TH_Circ_A, TH_Circ_a, TH_Circ_B, TH_Ratio_B, TH_Circ_C, TH_Ratio_C); //parametro per calcolare pi� o meno ccerhi // 10 <parametro< 20
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

	  //design_Circles(total, images[i]);
  }

  clock_t end = clock();

  cv::waitKey(0);
  cv::destroyAllWindows();

  total_time = (float)(end - start) / CLOCKS_PER_SEC * 1000;
  cout << "TOTAL TIME TOTAL TIME: " << total_time << "ms" << endl;

  //ALI YOU SHPULD START FROM HERE

	  //for visualize balls with squared boxes: design_Boxes
	  //for calculate the coordinates of squared boxes: calculate_SquaresVertices
  for (int i = 0; i < size; i++) {
	  vector < vector<cv::Point2f>> vertices_boxes = calculate_SquaresVertices(total_circles[i]);
	  design_Boxes(vertices_boxes, images[i]);
  }
  /*for (int index = 0; index < 10; index++) {
    clock_t start = clock();

    Mat field_mask = Mat::zeros(in.rows, in.cols, CV_8UC3);
    Vec4Points vertices = detect_field(in, field_mask);
    clock_t end = clock();
    // cout << "time" << (float)(end - start) / CLOCKS_PER_SEC * 1000 << "ms"
    //     << endl;
    circle(in, vertices[0], 20, Scalar(255, 100, 255), LINE_8);
    circle(in, vertices[1], 20, Scalar(255, 100, 255), LINE_8);
    circle(in, vertices[2], 20, Scalar(255, 100, 255), LINE_8);
    circle(in, vertices[3], 20, Scalar(255, 100, 255), LINE_8);
    imshow("Vertices", in);
    imshow("Mask", field_mask);
    waitKey(0);
    total_time += (float)(end - start) / CLOCKS_PER_SEC * 1000;
    // imshow("field", field);
  }*/
  cout << total_time << endl;
  return 0;
}
