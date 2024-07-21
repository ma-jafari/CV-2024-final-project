/*Author: Matteo De Gobbi */
#include "ball_classification.hpp"
#include "field_detection.hpp"
#include "minimap.hpp"
#include "opencv2/highgui.hpp"
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

// returns the path to the mp4 clip stored inside folderpath
std::string find_mp4_video(std::string folderpath) {
  namespace fs = std::filesystem;

  for (const auto &elem : fs::directory_iterator(folderpath)) {
    if (elem.path().extension() == ".mp4") {
      return elem.path().string();
    }
  }
  return "";
}
void track_balls(std::string path, std::vector<cv::Rect> &bboxes,
                 std::vector<ball_class> &ball_classes, bool savevideo,
                 std::string out_savepath, Vec4Points table_vertices) {
  using namespace cv;
  std::string filename = find_mp4_video(path);

  VideoCapture cap(filename);
  if (!cap.isOpened()) {
    std::cout << "Cannot open the video file. \n";
    return;
  }
  Mat frame;
  cap >> frame;
  int minimap_w = frame.cols / 3;
  int minimap_h = frame.rows / 3;
  std::vector<Ptr<Tracker>> trackers;
  for (const auto &bbox : bboxes) {
    Ptr<Tracker> tracker = TrackerCSRT::create();
    tracker->init(frame, bbox);
    trackers.push_back(tracker);
  }
  VideoWriter writer;
  if (savevideo) {
    writer = VideoWriter(out_savepath + "billiard_output.avi",
                         VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                         Size(frame.cols, frame.rows));
  }

  // Matrix to store the trail of balls across frames
  Mat trailmap = Mat::zeros(minimap_h, minimap_w, CV_8UC3);

  while (cap.read(frame)) {
    if (frame.empty())
      break;

    for (size_t i = 0; i < trackers.size(); ++i) {
      bool isok = trackers[i]->update(frame, bboxes[i]);
      if (isok) {
        rectangle(frame, bboxes[i], Scalar(255, 255, 0), 2, LINE_4);
      } else {
        bboxes.erase(bboxes.begin() + i);
        trackers.erase(trackers.begin() + i);
      }
    }

    if (savevideo) {
      writer.write(frame);
    }
    Mat minimap;

    drawMinimap(bboxes, table_vertices, ball_classes, minimap, trailmap,
                minimap_w, minimap_h);
    // we draw the map in the bottom left corner of the frame
    int x = 0;
    int y = frame.rows - minimap.rows;
    cv::Rect minimap_roi(x, y, minimap.cols, minimap.rows);
    minimap.copyTo(frame(minimap_roi));

    //   imshow("minimap", minimap);
    imshow("tracker", frame);
    // skip video if ESC is pressed
    if (waitKey(1) == 27)
      break;
  }
}
