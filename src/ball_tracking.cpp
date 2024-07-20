/*Author: Matteo De Gobbi */
#include "opencv2/highgui.hpp"
#include <cstddef>
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

void track_balls(std::string s, std::vector<cv::Rect> bboxes) {
  using namespace cv;
  std::string filename = "../data/" + s + "/" + s + ".mp4";
  VideoCapture cap(filename);
  if (!cap.isOpened()) {
    std::cout << "Cannot open the video file. \n";
  }
  Mat frame;
  cap >> frame;

  std::vector<Ptr<Tracker>> trackers;
  for (const auto &bbox : bboxes) {
    Ptr<Tracker> tracker = TrackerCSRT::create();
    tracker->init(frame, bbox);
    trackers.push_back(tracker);
  }

  /*VideoWriter video("outcpp.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                    Size(frame.cols, frame.rows));
  */
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

    // video.write(frame);
    imshow("tracker", frame);

    if (waitKey(1) == 27)
      break;
  }
}
