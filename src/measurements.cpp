#include <cassert>
#include <cstddef>
#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>
#include <vector>

#include "ball_classification.hpp"

using namespace std;
using namespace cv;

// Function to compute bounding boxes from detected balls
vector<Rect> compute_bboxes(const vector<Vec3f> &balls) {
  vector<Rect> bboxes;
  for (const auto &ball : balls) {
    int x = static_cast<int>(ball[0] - ball[2]);
    int y = static_cast<int>(ball[1] - ball[2]);
    int width = static_cast<int>(2 * ball[2]);
    int height = static_cast<int>(2 * ball[2]);
    bboxes.emplace_back(x, y, width, height);
  }
  return bboxes;
}

// Function to load ground truth and prediction data
void loadGroundTruthAndPredictions(const vector<vector<int>> &data,
                                   vector<Rect> &boxes) {
  for (const auto &boxData : data) {
    int x = boxData[0];
    int y = boxData[1];
    int width = boxData[2];
    int height = boxData[3];
    boxes.emplace_back(x, y, width, height);
  }
}

// Function to compute Intersection over Union (IoU)
double computeIoU(const Rect &gtBox, const Rect &predBox) {
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
// FIX: this function is wrong, it should be applied to segmentation masks not
// to bounding boxes
double computeMeanIoU(const vector<Rect> &gtBoxes,
                      const vector<Rect> &predBoxes) {
  double sumIoU = 0.0;
  int count = 0;
  for (const auto &gtBox : gtBoxes) {
    double maxIoU = 0.0;
    for (const auto &predBox : predBoxes) {
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
// Function to compute Precision and Recall for a single class specified by
// classID, the results are stored in the vector passed by reference
// incremental_precisions and incremental_recalls
void computePrecisionRecall(const vector<Rect> &gtBoxes,
                            const vector<Rect> &predBoxes,
                            const vector<ball_class> &gtClassIDs,
                            const vector<ball_class> &predClassIDs,
                            vector<double> &incremental_precisions,
                            vector<double> &incremental_recalls,
                            ball_class classID) {

  constexpr double iouThreshold = 0.5; // fixed threshold for Pascal VOC mAP

  incremental_precisions.clear();
  incremental_recalls.clear();

  // to keep track of which ground truth bboxes are already matched
  vector<bool> gtMatched(gtBoxes.size(), false);
  int tp = 0; // True positives
  int fp = 0; // False positives

  for (size_t i = 0; i < predBoxes.size(); i++) {
    double maxiou = -1.0;
    int maxiou_index = -1;
    if (predClassIDs[i] != classID) {
      continue; // skip this predBox, we are only computing AP for class classID
    }
    for (size_t j = 0; j < gtBoxes.size(); j++) {
      if ((gtClassIDs[j] != classID) || gtMatched[j]) {
        continue;
        // skip this gtBox, we are only computing AP for class classID
        // and at most one match for each gtBox
      }
      double iou = computeIoU(gtBoxes[j], predBoxes[i]);
      cout << "iou" << iou << endl;
      if (iou > maxiou) {
        // if we find a better fitting gt box we save it and its corresponding
        // IoU
        maxiou = iou;
        maxiou_index = j;
      }
    }
    cout << "maxiou" << maxiou << endl;
    if (maxiou >= iouThreshold) {
      ++tp;
      // true positive, both IoU is above threshold and prediction is correct
      gtMatched[maxiou_index] = true; // this gt box cannot be used anymore
    } else {
      // false positive, either the wrong class or IoU is too small
      ++fp;
    }
    cout << "tp" << tp << endl;
    cout << "fp" << fp << endl;
    incremental_precisions.push_back(static_cast<double>(tp) / (tp + fp));
    incremental_recalls.push_back(static_cast<double>(tp) / gtBoxes.size());
  }
}

// Function to compute Average Precision (AP) using Pascal VOC 11-point
// interpolation of the class specified by classID
double computeAP(const vector<Rect> &gtBoxes, const vector<Rect> &predBoxes,
                 vector<ball_class> &gtClassIDs,
                 vector<ball_class> &predClassIDs, ball_class classID) {

  // incremental recalls and precisions by considering an increasing amount of
  // bounding boxes
  vector<double> precisions;
  vector<double> recalls;
  computePrecisionRecall(gtBoxes, predBoxes, gtClassIDs, predClassIDs,
                         precisions, recalls, classID);

  cout << "Precisions:";
  for (auto &el : precisions) {
    cout << el << ",";
  }
  cout << endl;
  cout << "Recalls:";
  for (auto &el : recalls) {
    cout << el << ",";
  }
  cout << endl;

  // 11 levels where precision is interpolated according to pascal voc
  vector<double> recall_levels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                  0.6, 0.7, 0.8, 0.9, 1.0};

  size_t i = 0;
  double ap = 0.0;
  for (auto &recall_level : recall_levels) {
    while (i < recalls.size() && recalls[i] < recall_level) {
      ++i;
    }
    if (i >= recalls.size()) {
      break;
    }
    ap += precisions[i];
  }
  ap /= recall_levels.size();
  /*
  double ap = 0.0;
  std::vector<double> precision;
  std::vector<double> recall;

  // Compute precision-recall for each threshold
  for (auto &t : pascalPts) {
    auto [prec, rec] =
        computePrecisionRecall(gtBoxes, predBoxes,
  gtClassIDs, predClassIDs);
    precision.push_back(prec);
    recall.push_back(rec);
  }

  // Compute AP using 11-point interpolation
  for (size_t i = 0; i < pascalPts.size(); i++) {
    double max_precision = 0;
    for (size_t j = 0; j < recall.size(); j++) {
      if (recall[j] >= pascalPts[i] && precision[j]
  > max_precision) { max_precision = precision[j];
      }
    }
    ap += max_precision;
  }

  return ap / pascalPts.size();
  */
  return ap;
}

double computeMeanAP(const vector<Rect> &gtBoxes, const vector<Rect> &predBoxes,
                     vector<ball_class> &gtClassIDs,
                     vector<ball_class> &predClassIDs) {
  double solidAP = computeAP(gtBoxes, predBoxes, gtClassIDs, predClassIDs,
                             ball_class::SOLID);

  double stripedAP = computeAP(gtBoxes, predBoxes, gtClassIDs, predClassIDs,
                               ball_class::STRIPED);

  double cueAP =
      computeAP(gtBoxes, predBoxes, gtClassIDs, predClassIDs, ball_class::CUE);

  double eightballAP = computeAP(gtBoxes, predBoxes, gtClassIDs, predClassIDs,
                                 ball_class::SOLID);
  return (solidAP + stripedAP + cueAP + eightballAP);
}
