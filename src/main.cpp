#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <strings.h>

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
    clock_t start = clock();

    detect_field(in);
    clock_t end = clock();
    // cout << "time" << (float)(end - start) / CLOCKS_PER_SEC * 1000 << "ms"
    //     << endl;
    total_time += (float)(end - start) / CLOCKS_PER_SEC * 1000;
    // imshow("field", field);
  }
  cout << total_time << endl;
  return 0;
}
