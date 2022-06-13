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

void draw_histogram(const cv::Mat& img, const std::string &str)
{

	cv::Mat histogram;

	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 256;

	calcHist(&img, 1, channel_numbers, cv::Mat(), histogram, 1, &number_bins, &channel_ranges);

	std::cout << histogram.rows << " " << histogram.cols << std::endl; // 255, 1
	std::cout << histogram.type() << std::endl;
	std::cout << CV_32F << std::endl;
	int width = 256 * 3;
	int height = 700;
	cv::Mat plane = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);

	double minVal;
	double maxVal;
	cv::Point minLoc;
	cv::Point maxLoc;

	cv::minMaxLoc(histogram, &minVal, &maxVal, &minLoc, &maxLoc);
	std::cout << maxVal << " " << maxLoc <<std::endl;

	float* hist = &histogram.at<float>(0);
	int interval = 50;
	for (int i = 0; i < number_bins; ++i)
	{	
		cv::line(plane, cv::Point(i * width / number_bins, height - (hist[i] / maxVal * (height - interval))), cv::Point(i * width / number_bins, height), cv::Scalar(128, 200, 128), 1);
	}
	cv::resize(plane, plane, cv::Size(img.cols, img.rows));
	cv::imshow(str, plane);
}

void histogram_stretching(const cv::Mat& img, cv::Mat& res)
{
	double minVal;
	double maxVal;
	cv::Point minLoc;
	cv::Point maxLoc;

	cv::minMaxLoc(img, &minVal, &maxVal, &minLoc, &maxLoc);
	res = (img - minVal) / (maxVal - minVal) * 255;

	cv::imshow("histogram stretching", res);
}

void histogram_equalization(const cv::Mat& img, cv::Mat& res)
{
	cv::equalizeHist(img, res);
	cv::imshow("histogram equalization", res);
}

void add_image(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& res)
{
	res = src1 + src2;
	cv::imshow("add image", res);
}

void sub_image(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& res)
{
	res = src1 - src2;
	cv::imshow("sub image", res);
}

void avg_image(const std::vector<cv::Mat>& src, cv::Mat& res)
{
	int n = src.size();
	cv::Mat temp;

	res = cv::Mat::zeros(src[0].size(), CV_8UC1);
	for (int i = 0; i < n; ++i)
	{
		res += src[i] / n;
	}
	cv::imshow("avg image", res);
}

void abs_sub_image(cv::Mat& src1, cv::Mat& src2, cv::Mat& res)
{
	std::cout << "?" << std::endl;
	if (src1.cols != src2.cols || src1.rows != src2.rows)
		cv::resize(src2, src2, cv::Size(src1.cols, src2.rows));

	cv::absdiff(src1, src2, res);
	cv::imshow("abs sub image", res);
}

void and_image(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& res)
{
	cv::bitwise_and(src1, src2, res);

	cv::imshow("img2", src2);
	/*res = cv::Mat::zeros(cv::Size(src1.rows, src1.cols), CV_8UC1);
	for (int i = 0; i < src1.rows; ++i)
	{
		for (int j = 0; j < src1.cols; ++j)
		{
			res.data[i * res.cols + j] = src1.data[i * src1.cols + j] & src2.data[i * src2.cols + j];
		}
	}*/
	cv::imshow("and image", res);
}

void or_image(const cv::Mat& src1, const cv::Mat& src2, cv::Mat& res)
{	
	//cv::bitwise_or(src1, src2, res);

	cv::imshow("img2", src2);
	res = cv::Mat::zeros(cv::Size(src1.rows, src1.cols), CV_8UC1);
	for (int i = 0; i < src1.rows; ++i)
	{
		for (int j = 0; j < src1.cols; ++j)
		{
			res.data[i * res.cols + j] = src1.data[i * src1.cols + j] | src2.data[i * src2.cols + j];
		}
	}
	cv::imshow("or image", res);

}

void bit_plane(const cv::Mat& src, cv::Mat& res, int bit)
{
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			res.data[i * res.cols + j] = (src.data[i * src.cols + j] & (1 << bit)) ? 255 : 0;
		}
	}
	cv::imshow("bit plane " + std::to_string(bit) + " image" , res);
}

void average_filter(const cv::Mat& src, cv::Mat& res, const unsigned int ksize)
{
	//cv::blur(src, res, cv::Size(ksize, ksize));
	cv::Mat kernel = cv::Mat::ones(cv::Size(ksize, ksize), CV_32F);
	kernel /= ksize * ksize;
	cv::filter2D(src, res, -1, kernel);
	std::string window_name = "평균값 필터 " + std::to_string(ksize);
	cv::imshow(window_name, res);
}

void weighted_filter(const cv::Mat& src, cv::Mat& res)
{
	float data[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	cv::Mat kernel = cv::Mat(3, 3, CV_32F, data);
	kernel /= cv::sum(kernel);
	cv::filter2D(src, res, -1, kernel);
	std::string window_name = "가중 평균값 필터";
	cv::imshow(window_name, res);
}

void gaussian(const cv::Mat& src, cv::Mat& res, const unsigned kernel_size, const double sigmaX, const double sigmaY)
{
	cv::GaussianBlur(src, res, cv::Size(kernel_size, kernel_size), sigmaX, sigmaY);
	std::string window_name = "가우시안 필터 적용 " + std::to_string(kernel_size) + " " + std::to_string(sigmaX) + " " + std::to_string(sigmaY);
	cv::imshow(window_name, res);
	
}