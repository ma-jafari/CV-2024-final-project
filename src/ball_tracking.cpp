/*Author: Matteo De Gobbi */
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

std::string find_mp4_video(std::string folderpath) {
  namespace fs = std::filesystem;

  for (const auto &elem : fs::directory_iterator(folderpath)) {
    if (elem.path().extension() == ".mp4") {
      return elem.path().string();
    }
  }
  return "";
}
void track_balls(std::string path, std::vector<cv::Rect> bboxes, bool savevideo,
                 std::string out_savepath) {
  using namespace cv;
  std::string filename = find_mp4_video(path);

  VideoCapture cap(filename);
  if (!cap.isOpened()) {
    std::cout << "Cannot open the video file. \n";
    return;
  }
  Mat frame;
  cap >> frame;

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

  while (true) {
    cap >> frame;
    if (frame.empty())
      break;

    for (size_t i = 0; i < trackers.size(); ++i) {
      bool isok = trackers[i]->update(frame, bboxes[i]);
      if (isok) {
        rectangle(frame, bboxes[i], Scalar(255, 255, 0), 2, LINE_4);
      }
    }

    if (savevideo) {
      writer.write(frame);
    }
    imshow("tracker", frame);

    if (waitKey(1) == 27)
      break;
  }
}
