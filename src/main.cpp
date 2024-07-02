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

#include "ball_classification.hpp"
#include "ball_detection.hpp"
#include "field_detection.hpp"
#include "functions.h"

using namespace cv;
using namespace std;

int main() {
  string base_path = "../data/";
  string names[] = {"game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4",
                    "game2_clip1", "game2_clip2", "game3_clip1", "game3_clip2",
                    "game4_clip1", "game4_clip2"};

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

  for (auto &in_img : images) {
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
    // draw_bboxes(vertices_boxes, in_img);
    circle(cutout_table, vertices[0], 20, Scalar(0, 0, 255));
    circle(cutout_table, vertices[1], 20, Scalar(0, 0, 255));
    circle(cutout_table, vertices[2], 20, Scalar(0, 0, 255));
    circle(cutout_table, vertices[3], 20, Scalar(0, 0, 255));
    imshow("vertices", cutout_table);

    //

    // Scalar FColor = computeDominantColor(images[i]);
    // cout << "dominante color: " << FColor << endl;

    // Classify balls within the boxes
    for (const auto &box : vertices_boxes) {
      Rect rect(box[0] - Point2f(5, 5),
                box[2] + Point2f(5, 5)); // Assuming box[0] is top-left and
                                         // box[2] is bottom-right
      Mat roi = in_img(rect);
      Mat ballroi = detectBalls(roi);

      namedWindow("test");

      resizeWindow("test", 400, 300);
      imshow("test", ballroi);
      // Classify the ball using adaptive thresholding
      if (classify_ball(ballroi) == ball_class::STRIPED) {
        rectangle(in_img, rect, Scalar(0, 255, 0),
                  2); // Green for striped balls
      } else {
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
