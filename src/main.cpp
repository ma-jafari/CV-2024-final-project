#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "field_detection.hpp"

int main() {
  using namespace cv;
  using namespace std;
  string names[] = {"game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4",
                    "game2_clip1", "game2_clip2", "game3_clip1", "game3_clip2",
                    "game4_clip1", "game4_clip2"};

  double total_time = 0.0;
  for (int index = 0; index < 10; index++) {
    Mat in = cv::imread(string("data/") + names[index] +
                        string("/frames/frame_first.png"));
    if (in.empty()) {
      cout << "Error file not found: " << names[index] << endl;
      return -1;
    }
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
  }
  cout << total_time << endl;
  return 0;
}
