#include "image_utils.h"
void image_read(const char* file_path, cv::Mat &img, const int flag)
{
	img = cv::imread(file_path, flag);
	cv::imshow("img", img);
}

void brightness_inverse(const cv::Mat& img, cv::Mat& res)
{
	res = 255 - img;
	cv::imshow("inverse img", res);
}

void add_scalar(const cv::Mat& img, const int k, cv::Mat& res)
{
	res = img + k;
	cv::imshow("add scalar", res);
}

void multiply_scalar(const cv::Mat& img, const float k, cv::Mat& res)
{
	res = img * k;
	cv::imshow("multiply scalar", res);
}

void improve_contrast(const cv::Mat& img, const float k, cv::Mat& res)
{
	res = img + ((img - 128) * k);
	cv::imshow("improve_contrast image2", res);
}

void gamma_correction(const cv::Mat& img, const float k, cv::Mat& res)
{
	img.convertTo(res, CV_32FC1, 1. / 255.);
	cv::pow(res, 1. / k, res);
	res.convertTo(res, CV_8UC1, 255);
	cv::imshow("gamma correction", res);
}