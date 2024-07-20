#include <algorithm>
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
#include "field_detection.hpp"
#include "functions.h"
#include "measurements.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  const String keys =
      "{help h usage ? | | print the help message }"
      "{i intermidiate| | show intermediate steps of the algorithm}"
      "{s save | | save the output video on a file}"
      "{savepath | | path of the output video when -s or --save are used}"
      "{b benchmark| | use this argument if the path provided contains "
      "multiple subdirectories containing clips, in this case the program will "
      "compute the accuracy metrics across all the dataset}"
      "{@path|../data/game1_clip1 | path to the dataset or directory of the "
      "clip}";
  CommandLineParser parser(argc, argv, keys);
  parser.about("Application name v1.0.0");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  bool show_intermediate = parser.has("i");
  if (parser.has("benchmark")) {
    cout << "Not implemeneted yet" << endl;
    return -1;
  } else if (true) {
    string path = parser.get<string>("@path");
    // cout << path << endl;

    string imagePath = path + "/frames/frame_first.png";
    Mat first_frame = imread(imagePath);
    if (first_frame.empty()) {
      cerr << "Error loading image file: " << imagePath << endl;
      return -1;
    }

    Mat cutout_table;
    Mat mask = Mat::zeros(first_frame.rows, first_frame.cols, CV_8UC3);

    // NOTE: field detection
    // NOTE: We cut find the table boundaries and cut out the table
    // from the rest of the image
    Vec4Points vertices = detect_field(first_frame);
    fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
    bitwise_and(first_frame, mask, cutout_table);

    Scalar linecolor = Scalar(255, 0, 0);
    int linewidth = LINE_4;
    line(cutout_table, vertices[0], vertices[1], linecolor, linewidth);
    line(cutout_table, vertices[2], vertices[1], linecolor, linewidth);
    line(cutout_table, vertices[2], vertices[3], linecolor, linewidth);
    line(cutout_table, vertices[3], vertices[0], linecolor, linewidth);
    imshow("out", cutout_table);
    Mat cutout_original = cutout_table.clone();
    imshow("mask", mask);

    // NOTE: ball detection
    //  NOTE: remove balls on edge of table
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

    // draw_bboxes(vertices_boxes, in_img);
    circle(cutout_table, vertices[0], 20, Scalar(0, 0, 255));
    circle(cutout_table, vertices[1], 20, Scalar(0, 0, 255));
    circle(cutout_table, vertices[2], 20, Scalar(0, 0, 255));
    circle(cutout_table, vertices[3], 20, Scalar(0, 0, 255));
    imshow("vertices", cutout_table);

    vector<vector<cv::Point2f>> vertices_boxes =
        compute_bbox_vertices(selected_balls);
    Mat classifiedImg = first_frame.clone();
    vector<cv::Rect> bbox_rectangles = compute_bboxes(selected_balls);

    // NOTE: ball classification
    vector<ball_class> pred_classes(vertices_boxes.size());

    // vector<cv::Rect> bbox_rectangles;
    // Classify balls within the boxes
    for (int j = 0; j < vertices_boxes.size(); j++) {
      vector<Point2f> box = vertices_boxes[j];
      Rect rect(box[0],
                box[2]); // Assuming box[0] is top-left and
                         // box[2] is bottom-right
      // bbox_rectangles.push_back(rect);
      Mat roi = first_frame(rect);

      // namedWindow("test");

      // resizeWindow("test", 400, 300);
      // imshow("test", ballroi);
      ball_class classifiedBall = classify_ball(roi);
      if (classifiedBall == ball_class::STRIPED) {
        rectangle(classifiedImg, rect, Scalar(0, 255, 0),
                  2); // Green for striped balls
      } else if (classifiedBall == ball_class::SOLID) {
        rectangle(classifiedImg, rect, Scalar(0, 0, 255),
                  2); // Red for solid balls
      } else if (classifiedBall == ball_class::CUE) {

        rectangle(classifiedImg, rect, Scalar(255, 255, 255),
                  2); // white for cue balls
      } else {
        rectangle(classifiedImg, rect, Scalar(0, 0, 0),
                  2); // black for 8balls
      }
      pred_classes[j] = classifiedBall;
    }
    imshow("classified image", classifiedImg);

    vector<Rect> gtBoxes;   // Ground truth bounding boxes for this image
    vector<Rect> predBoxes; // Predicted bounding boxes for this image
    vector<vector<int>> labels;
    string labelPath = path + "/bounding_boxes/frame_first_bbox.txt";
    bool success_label_reading = extractLabelsFromFile(labelPath, labels);
    vector<ball_class> gt_classes(labels.size());
    if (success_label_reading) {
      for (int j = 0; j < labels.size(); j++) {
        gt_classes[j] = int2ball_class(labels[j][4]);
      }
    } else {
      cerr << "Failed to find all required labels in file: " << labelPath
           << endl;
    }

    // Load ground truth and predictions
    loadGroundTruthAndPredictions(labels, gtBoxes);
    predBoxes = bbox_rectangles;
    Mat Bboxes_img = first_frame.clone();

    // Visualize ground truth and predicted boxes
    for (const auto &gtBox : gtBoxes) {
      rectangle(Bboxes_img, gtBox, Scalar(0, 0, 255),
                2); // Red rectangles for ground truth boxes
    }
    for (const auto &predBox : predBoxes) {
      rectangle(Bboxes_img, predBox, Scalar(255, 0, 0),
                2); // Blue rectangles for predicted boxes
    }

    // imshow("Bounding Boxes", Bboxes_img);

    //    cout << "Solid ball average precision" << averagePrecision << endl;

    cout << computeMeanAP(gtBoxes, predBoxes, gt_classes, pred_classes) << endl;
    waitKey();
  }
  return 0;
}
