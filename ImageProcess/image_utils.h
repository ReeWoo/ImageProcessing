#pragma once
#include "opencv2/opencv.hpp"

void image_read(const char* file_path, cv::Mat &img, const int flag = 1);
void brightness_inverse(const cv::Mat& img, cv::Mat& res);
void add_scalar(const cv::Mat& img, const int k, cv::Mat& res);
void multiply_scalar(const cv::Mat& img, const float k, cv::Mat& res);
void improve_contrast(const cv::Mat& img, const float k, cv::Mat& res);