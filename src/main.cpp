#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "ball_detection.hpp"
#include "field_detection.hpp"

int main() {
  using namespace cv;
  using namespace std;
  string names[] = {"game1_clip1", "game1_clip2", "game1_clip3", "game1_clip4",
                    "game2_clip1", "game2_clip2", "game3_clip1", "game3_clip2",
                    "game4_clip1", "game4_clip2"};

  for (int index = 0; index < 10; index++) {
    Mat in = cv::imread(string("data/") + names[index] +
                        string("/frames/frame_first.png"));
    Mat masked_in = in.clone();
    if (in.empty()) {
      cout << "Error file not found: " << names[index] << endl;
      return -1;
    }
    Mat field_mask = Mat::zeros(in.rows, in.cols, CV_8UC3);
    Vec4Points vertices = detect_field(in, field_mask);

    Vec2i center = (vertices[0] + vertices[1] + vertices[2] + vertices[3]) / 4;

    bitwise_and(in, field_mask, masked_in);
    circle(in, vertices[0], 20, Scalar(255, 100, 255), LINE_8);
    circle(in, vertices[1], 20, Scalar(255, 100, 255), LINE_8);
    circle(in, vertices[2], 20, Scalar(255, 100, 255), LINE_8);
    circle(in, vertices[3], 20, Scalar(255, 100, 255), LINE_8);
    circle(in, center, 20, Scalar(255, 100, 255), LINE_8);
    imshow("Vertices", in);

    // imshow("masked", masked_in);

    // imshow("Mask", field_mask);
    detect_balls(masked_in);
    waitKey(0);
  }
  return 0;
}
