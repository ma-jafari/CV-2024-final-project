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
#include "ball_tracking.hpp"
#include "field_detection.hpp"
#include "measurements.hpp"
#include "minimap.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  const String keys =
      "{help h usage ? | | print the help message }"
      "{i intermidiate| | show intermediate steps of the algorithm}"
      "{s save | | save the output video on a file}"
      "{savepath | | path of the directory where the output video wil be "
      "saved-s or --save are used}"
      "{@path|../data/game1_clip1 | path to the directory of the clip}";
  CommandLineParser parser(argc, argv, keys);
  parser.about("Billiard Analisys");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }
  bool show_intermediate = parser.has("i");
  bool savevideo = parser.has("s");
  string savepath = savevideo ? parser.get<string>("savepath") : "";
  string path = parser.get<string>("@path");

  vector<string> frameAndMasksNames = {"frame_first.png", "frame_last.png"};
  vector<Mat> frames;
  vector<Mat> gtMasks;

  for (const auto &name : frameAndMasksNames) {
    string imagePath = path + "/frames/" + name;
    Mat frame = imread(imagePath);
    if (frame.empty()) {
      cerr << "Error loading image file: " << imagePath << endl;
      return -1;
    }
    frames.push_back(frame);

    string maskPath = path + "/masks/" + name;
    Mat gtMask = imread(maskPath, IMREAD_GRAYSCALE);
    if (gtMask.empty()) {
      cerr << "Error loading mask file: " << maskPath << endl;
      return -1;
    }
    gtMasks.push_back(gtMask);
  }

  // NOTE: field detection
  Mat mask = Mat::zeros(frames[0].rows, frames[0].cols, CV_8UC3);
  Vec4Points vertices = detect_field(frames[0], show_intermediate);
  fillPoly(mask, vertices, cv::Scalar(255, 255, 255));
  for (int k = 0; k < frames.size(); k++) {
    Mat frame = frames[k];
    Mat gtMask = gtMasks[k];
    vector<Rect> predBoxes;
    vector<int> predClassIds;
    vector<double> meanIoUPerClass(6, 0.0);
    vector<int> classIoUCount(6, 0);

    Mat cutout_table;

    // NOTE: We cut find the table boundaries and cut out the table
    // from the rest of the image
    bitwise_and(frame, mask, cutout_table);

    cut_table(frame, mask, vertices, cutout_table);

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

    vector<vector<cv::Point2f>> vertices_boxes =
        compute_bbox_vertices(selected_balls);
    Mat classifiedImg = frame.clone();
    vector<cv::Rect> bbox_rectangles = compute_bboxes(selected_balls);

    // NOTE: ball classification
    vector<ball_class> pred_classes(vertices_boxes.size());

    // Vectors to store circles for each class
    vector<Point2f> stripped_balls;
    vector<Point2f> solid_balls;
    vector<Point2f> white_balls;
    vector<Point2f> black_balls;

    // Masks for each class
    Mat mask_stripped = Mat::zeros(frame.size(), CV_8UC1);
    Mat mask_solid = Mat::zeros(frame.size(), CV_8UC1);
    Mat mask_white = Mat::zeros(frame.size(), CV_8UC1);
    Mat mask_black = Mat::zeros(frame.size(), CV_8UC1);

    // vector<cv::Rect> bbox_rectangles;
    // Classify balls within the boxes
    for (int j = 0; j < vertices_boxes.size(); j++) {
      vector<Point2f> box = vertices_boxes[j];
      Rect rect(box[0],
                box[2]); // Assuming box[0] is top-left and
                         // box[2] is bottom-right
      Mat roi = frame(rect);

      ball_class classifiedBall = classify_ball(roi);
      Point2f center = (box[0] + box[2]) * 0.5;
      float radius = (norm(box[0] - box[2]) * 0.5) * 2 / 3;

      Scalar ball_class_color = ball_class2color(classifiedBall);
      if (classifiedBall == ball_class::STRIPED) {
        rectangle(classifiedImg, rect, ball_class_color, 2);
        stripped_balls.push_back(center);
        circle(mask_stripped, center, radius, Scalar(255), -1);
      } else if (classifiedBall == ball_class::SOLID) {
        rectangle(classifiedImg, rect, ball_class_color, 2);
        solid_balls.push_back(center);
        circle(mask_solid, center, radius, Scalar(255), -1);
      } else if (classifiedBall == ball_class::CUE) {
        rectangle(classifiedImg, rect, ball_class_color,
                  2); // White for white ball
        white_balls.push_back(center);
        circle(mask_white, center, radius, Scalar(255), -1);
      } else if (classifiedBall == ball_class::EIGHT_BALL) {
        rectangle(classifiedImg, rect, ball_class_color,
                  2); // Black for black ball
        black_balls.push_back(center);
        circle(mask_black, center, radius, Scalar(255), -1);
      }
      pred_classes[j] = classifiedBall;
    }
    imshow(frameAndMasksNames[k] + " classified image", classifiedImg);

    vector<Mat> ballMasks = {mask_white, mask_black, mask_solid, mask_stripped};
    ComputeMeanIoU(frame, gtMask, vertices, path, ballMasks);

    predBoxes = bbox_rectangles;

    vector<Rect> gtBoxes; // Ground truth bounding boxes for this image
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
    Mat Bboxes_img = frame.clone();
    cout << "mAP of " << frameAndMasksNames[k] << " "
         << computeMeanAP(gtBoxes, predBoxes, gt_classes, pred_classes) << endl;
    // we only do the tracking one the first frame
    if (k == 0) {
      track_balls(path, predBoxes, pred_classes, savevideo, savepath, vertices);
    }
    waitKey(0);
  }
  return 0;
}
