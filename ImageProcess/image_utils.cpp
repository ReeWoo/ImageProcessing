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