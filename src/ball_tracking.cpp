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
  printf("OpenCV: %s", cv::getBuildInformation().c_str());
  std::string filename = "../data/" + s + "/" + s + ".mp4";
  VideoCapture cap(filename);
  if (!cap.isOpened()) {
    std::cout << "Cannot open the video file. \n";
  }

  Ptr<cv::legacy::MultiTracker> multiTracker =
      cv::legacy::MultiTracker::create();

  // Initialize multitracker
  Mat frame;
  cap >> frame;
  for (int i = 0; i < bboxes.size(); ++i) {
    multiTracker->add(cv::legacy::TrackerCSRT::create(), frame,
                      Rect2d(bboxes[i]));
  }
  // Initialize multitracker
  while (true) {
    // get frame from the video
    cap >> frame;

    if (frame.rows == 0 || frame.cols == 0)
      break;

    multiTracker->update(frame);
    // draw the tracked object
    for (size_t i = 0; i < multiTracker->getObjects().size(); ++i) {
      rectangle(frame, multiTracker->getObjects()[i], Scalar(255, 255, 0));
    }

    imshow("tracker", frame);
    // quit on ESC button
    if (waitKey(1) == 27)
      break;
  }
}
