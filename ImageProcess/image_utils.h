#pragma once
#include "opencv2/opencv.hpp"

void image_read(const char* file_path, cv::Mat &img, const int flag = 1);