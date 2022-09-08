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
void sharpening(const cv::Mat& src, cv::Mat& res);
void laplacian(const cv::Mat& src, cv::Mat& res);
void high_boost(const cv::Mat& src, cv::Mat& res, float alpha);
void add_gaussian_noise(const cv::Mat& src, cv::Mat& res, float mean, float std, int percentage);
void salt_pepper(const cv::Mat& src, cv::Mat& res, float percentage);
void median_filter(const cv::Mat& src, cv::Mat& res);
void translation(const cv::Mat& src, cv::Mat& res, const int alpha, const int beta);
void scale_transform(const cv::Mat& src, cv::Mat& res, const float alpha, const float beta);
void scale_transform_with_bilinear(const cv::Mat& src, cv::Mat& res, const float alpha, const float beta);
void scale_transform_3order_interpolation(const cv::Mat& src, cv::Mat& res, const float alpha, const float beta);
void edge_detection_roberts(const cv::Mat& src, cv::Mat& res);
void edge_detection_prewitt(const cv::Mat& src, cv::Mat& res);
void edge_detection_sobel(const cv::Mat& src, cv::Mat& res);
void edge_detection_canny(const cv::Mat& src, cv::Mat& res, int threshold1, int threshold2);
void on_threshold1_change(int pos, void* userdata);
void on_threshold2_change(int pos, void* userdata);
void hough_transform(const cv::Mat& src, cv::Mat& res);
std::vector<cv::Point2f> harris_corner_detection(const cv::Mat& src, cv::Mat& res);

void binarization(const cv::Mat& src, cv::Mat& res, unsigned char T);
void calc_histogram(const cv::Mat& src, std::vector<int> histogram);
void iterative_binarization(const cv::Mat& src, cv::Mat& res, unsigned char T = 0);

void image_labeling(const cv::Mat& src, cv::Mat &res, std::vector<int> table);
void contour_tracing(const cv::Mat& src, int sx, int sy, std::vector<cv::Point2i> &vec);
void contour_trace(const cv::Mat& src, cv::Mat& res);

void color_inverse(const cv::Mat& src, cv::Mat& res);
void color2gray(const cv::Mat& src, cv::Mat& res);
void color_edge_detection(const cv::Mat& src, cv::Mat& res);
void color_histogram_equalization(const cv::Mat& src, cv::Mat& res);

void erosion(const cv::Mat& src, cv::Mat& res);
void dilation(const cv::Mat& src, cv::Mat& res);

void opening(const cv::Mat& src, cv::Mat& res);
void closing(const cv::Mat& src, cv::Mat& res);

void contour_with_erosion(const cv::Mat& src, cv::Mat& res);

void matching_template(const cv::Mat& src, cv::Mat& res, cv::Mat &color_img);