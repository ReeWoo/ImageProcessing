#pragma once
#include "opencv2/opencv.hpp"

void image_read(const char* file_path, cv::Mat &img, const int flag = 1);
void brightness_inverse(const cv::Mat& img, cv::Mat& res);
void add_scalar(const cv::Mat& img, const int k, cv::Mat& res);
void multiply_scalar(const cv::Mat& img, const float k, cv::Mat& res);
void improve_contrast(const cv::Mat& img, const float k, cv::Mat& res);
void gamma_correction(const cv::Mat& img, const float k, cv::Mat& res);
void draw_histogram(const cv::Mat& img, const std::string &str);
void histogram_stretching(const cv::Mat& img, cv::Mat& res);
void histogram_equalization(const cv::Mat& img, cv::Mat& res);
void add_image(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& res);
void sub_image(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& res);
void avg_image(const std::vector<cv::Mat> &src, cv::Mat& res);
void abs_sub_image(cv::Mat& src1, cv::Mat& src2, cv::Mat& res);
void and_image(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& res);
void or_image(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& res);
void bit_plane(const cv::Mat& src, cv::Mat& res, int bit);
void average_filter(const cv::Mat& src, cv::Mat& res, const unsigned int ksize);
void weighted_filter(const cv::Mat& src, cv::Mat& res);
void gaussian(const cv::Mat& src, cv::Mat& res, const unsigned kernel_size, const double sigmaX, const double sigmaY);