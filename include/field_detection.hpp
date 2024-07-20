/*Author: Matteo De Gobbi */
#ifndef FIELD_DETECTION_H
#define FIELD_DETECTION_H

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
typedef cv::Vec<cv::Point2i, 4> Vec4Points;
Vec4Points detect_field(const cv::Mat &input_image);

#endif // FIELD_DETECTION_H
