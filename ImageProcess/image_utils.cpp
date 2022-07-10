#include <random>
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

void sharpening(const cv::Mat& src, cv::Mat& res)
{
	cv::Mat smooth_img;
	average_filter(src, smooth_img, 3);
	cv::Mat g = src - smooth_img;
	res = g + src;
	cv::imshow("g" , g);
	cv::imshow("언샤프닝 마스크 필터링", res);
}

void laplacian(const cv::Mat& src, cv::Mat& res)
{
	
	//char data[] = { 0, -1, 0, -1, 5, -1, 0, -1, 0 };
	cv::Mat res2;

	char data[] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
	cv::Mat filter(3, 3, CV_8SC1, data);
	cv::filter2D(src, res2, -1, filter);
	cv::Laplacian(src, res, -1, 3);
	res = src - res;
	cv::imshow("라플라시안 필터링 opencv", res);
	cv::imshow("라플라시안 필터링", res2);
}

void high_boost(const cv::Mat& src, cv::Mat& res, float alpha)
{
	cv::Mat res2;

	cv::Laplacian(src, res2, -1, 3);
	res = alpha * src - res2;

	cv::imshow("하이부스트 " + std::to_string(alpha), res);
}

void add_gaussian_noise(const cv::Mat& src, cv::Mat& res, float mean, float std, int percentage)
{
	res = cv::Mat(cv::Size(src.cols, src.rows), src.type());

	cv::randn(res, mean, 100);
	res = src + res;


	/*unsigned int seed = static_cast<unsigned int>(time(nullptr));
	std::default_random_engine gen(seed);
	std::normal_distribution<double> normal_dis(mean, std);

	double random_val;
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			random_val = normal_dis(gen) * 255 * percentage / 100 + src.data[i * src.cols + j];
			random_val = random_val > 255 ? 255 : random_val;
			random_val = random_val < 0 ? 0 : random_val;
			res.data[i * res.cols + j] = static_cast<unsigned int>(random_val);
		}
	}*/
	
	cv::imshow("잡음 추가 평균 : " + std::to_string(mean) + " 표준편차 : " + std::to_string(std), res);
	


}

void salt_pepper(const cv::Mat& src, cv::Mat& res, float percentage)

{
	src.copyTo(res);

	int i, j;
	srand((int)time(NULL));
	int n = src.cols * src.rows * percentage / 100;
	for (int k = 0; k < n; ++k)

	{
		j = rand() % src.cols;
		i = rand() % src.rows;
		
		res.data[i * res.cols + j] = (rand() % 2) * 255;

	}
	cv::imshow("salt pepper 잡음 추가 비율 : " + std::to_string(percentage), res);
}

void median_filter(const cv::Mat& src, cv::Mat& res)
{	
	cv::Mat res2;
	
	salt_pepper(src, res2, 10);
	cv::medianBlur(res2, res, 3);

	cv::imshow("미디언 필터 적용 전", res2);
	cv::imshow("미디언 필터", res);
}

void translation(const cv::Mat& src, cv::Mat& res, const int alpha, const int beta)
{
	/*cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, alpha, 0, 1, beta);
	cv::warpAffine(src, res, trans_mat, cv::Size(src.cols + alpha, src.rows + beta));*/

	res = cv::Mat::zeros(cv::Size(src.cols + alpha, src.rows + beta), CV_8UC1);
	int x, y;
	for (int i = 0; i < res.rows; ++i)
	{
		for (int j = 0; j < res.cols; ++j)
		{
			x = j - alpha;
			y = i - beta;
			if (x < 0 || x >= src.cols || y < 0 || y >= src.rows)
				continue;
			res.data[i * res.cols + j] = src.data[y * src.cols + x];
		}
	}
	cv::imshow("이동 변환 ( " + std::to_string(alpha) + ", " + std::to_string(beta) + " )", res);
}

void scale_transform(const cv::Mat& src, cv::Mat& res, const float alpha, const float beta)
{
	res = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	int x, y;
	for (int i = 0; i < res.rows; ++i)
	{
		for (int j = 0; j < res.cols; ++j)
		{
			x = j / alpha;
			y = i / beta;
			if (x < 0 || x >= src.cols || y < 0 || y >= src.rows)
				continue;
			res.data[i * res.cols + j] = src.data[y * src.cols + x];
		}
	}
	/*for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			x = j * alpha;
			y = i * beta;
			if (x < 0 || x >= res.cols || y < 0 || y >= res.rows)
				continue;
			res.data[y * res.cols + x] = src.data[i * src.cols + j];
		}
	}*/
	cv::imshow("크기 변환 ( " + std::to_string(alpha) + ", " + std::to_string(beta) + " )", res);
}


void scale_transform_with_bilinear(const cv::Mat& src, cv::Mat& res, const float alpha, const float  beta)
{
	res = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	float x, y;
	int a, b, c, d, temp;
	float weight_x, weight_y;
	unsigned char x1, x2, x3, x4;
	for (int i = 0; i < res.rows; ++i)
	{
		for (int j = 0; j < res.cols; ++j)
		{
			x = j / alpha;
			y = i / beta;
			a = x;
			b = a + 1;
			c = y;
			d = c + 1;

			if (b >= src.cols) b = src.cols - 1;
			if (d >= src.rows) d = src.rows - 1;

			weight_x = x - a;
			weight_y = y - c;
			x1 = src.data[c * src.cols + a];
			x2 = src.data[c * src.cols + b];
			x3 = src.data[d * src.cols + a];
			x4 = src.data[d * src.cols + b];
			temp = ((1. - weight_y) * (weight_x* x2 + (1 - weight_x) * x1) + weight_y * (weight_x * x4 + (1 - weight_x) * x3));
			if (temp > 255) temp = 255;
			if (temp < 0) temp = 0;
			res.data[i * res.cols + j] = temp;

		}
	}
	cv::imshow("크기 변환 양선형보간 ( " + std::to_string(alpha) + ", " + std::to_string(beta) + " )", res);

}

float weighted_func(const float x)
{
	float res;
	float x_val = abs(x);
	if (x_val >= 0 && x_val < 1)
		res = 1.5 * x_val * x_val * x_val - 2.5 * x_val * x_val + 1;
	else if (x_val >= 1 && x_val < 2)
		res = -0.5 * x_val * x_val * x_val + 2.5 * x_val * x_val - 4 * x_val +2;
	else if (x_val >= 2)
		res = 0;
	return res;
}

void scale_transform_3order_interpolation(const cv::Mat& src, cv::Mat& res, const float alpha, const float beta)
{
	
	res = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_8UC1);
	float x, y;
	int a1, b1, a2, a3, a4, b2, b3, b4;
	int p1, p2, p3, p4;
	int p5, p6, p7, p8;
	int temp;
	for (int i = 0; i < res.rows; ++i)
	{
		for (int j = 0; j < res.cols; ++j)
		{
			x = j / alpha;
			y = i / beta;
			a1 = x;
			b1 = y;
			
			a2 = a1 - 1;
			a3 = a1 + 1;
			a4 = a1 + 2;
			b2 = b1 - 1;
			b3 = b1 + 1;
			b4 = b1 + 2;

			if (a2 < 0) a2 = 0;
			if (a3 >= src.cols) a3 = src.cols - 1;
			if (a4 >= src.cols) a4 = src.cols - 1;

			if (b2 < 0) b2 = 0;
			if (b3 >= src.rows) b3 = src.rows - 1;
			if (b4 >= src.rows) b4 = src.rows - 1;

			p1 = src.data[b2 * src.cols + a2];
			p2 = src.data[b2 * src.cols + a1];
			p3 = src.data[b2 * src.cols + a3];
			p4 = src.data[b2 * src.cols + a4];
			p5= weighted_func(x - a2)* p1 + weighted_func(x - a1) * p2 + weighted_func(a3 - x) * p3 + weighted_func(a4 - x) * p4;
			/*if (p5 > 255) p5 = 255;
			if (p5 < 0) p5 = 0;*/

			p1 = src.data[b1 * src.cols + a2];
			p2 = src.data[b1 * src.cols + a1];
			p3 = src.data[b1 * src.cols + a3];
			p4 = src.data[b1 * src.cols + a4];
			p6 = weighted_func(x - a2)* p1 + weighted_func(x - a1) * p2 + weighted_func(a3 - x) * p3 + weighted_func(a4 - x) * p4;
			/*if (p6 > 255) p6 = 255;
			if (p6 < 0) p6 = 0;*/

			p1 = src.data[b3 * src.cols + a2];
			p2 = src.data[b3 * src.cols + a1];
			p3 = src.data[b3 * src.cols + a3];
			p4 = src.data[b3 * src.cols + a4];
			p7 = weighted_func(x - a2)* p1 + weighted_func(x - a1) * p2 + weighted_func(a3 - x) * p3 + weighted_func(a4 - x) * p4;
			/*if (p7 > 255) p7 = 255;
			if (p7 < 0) p7 = 0;*/

			p1 = src.data[b4 * src.cols + a2];
			p2 = src.data[b4 * src.cols + a1];
			p3 = src.data[b4 * src.cols + a3];
			p4 = src.data[b4 * src.cols + a4];
			p8 = weighted_func(x - a2)* p1 + weighted_func(x - a1) * p2 + weighted_func(a3 - x) * p3 + weighted_func(a4 - x) * p4;
			/*if (p8 > 255) p8 = 255;
			if (p8 < 0) p8 = 0;*/

			temp = weighted_func(y - b2)* p5 + weighted_func(y - b1) * p6 + weighted_func(b3 - y) * p7 + weighted_func(b4 - y) * p8;
			if (temp > 255) temp = 255;
			if (temp < 0) temp = 0;
			res.data[i * res.cols + j] = temp;

		}
	}

	cv::imshow("크기 변환 3차 회선 보간 ( " + std::to_string(alpha) + ", " + std::to_string(beta) + " )", res);
}

void edge_detection_roberts(const cv::Mat& src, cv::Mat& res)
{	
	cv::Mat roberts1, roberts2;

	char data[] = { 0, 1, -1, 0 };
	char data2[] = { 1, 0, 0, -1 };
	cv::Mat filter(2, 2, CV_8SC1, data);
	cv::Mat filter2(2, 2, CV_8SC1, data2);
	cv::filter2D(src, roberts1, -1, filter);
	cv::filter2D(src, roberts2, -1, filter2);
	res = cv::abs(roberts1) + cv::abs(roberts2);


	cv::imshow("roberts filter 45 degree", roberts1);
	cv::imshow("roberts filter 135 degree", roberts2);
	
	cv::imshow("roberts", res);


}
void edge_detection_prewitt(const cv::Mat& src, cv::Mat& res)
{
	cv::Mat prewitt1, prewitt2;

	char data[] = { -1, 0,  1, -1, 0, 1, -1, 0, 1 };
	char data2[] = { -1, -1,  -1, 0, 0, 0, 1, 1, 1 };
	cv::Mat filter(3, 3, CV_8SC1, data);
	cv::Mat filter2(3, 3 , CV_8SC1, data2);
	cv::filter2D(src, prewitt1, -1, filter);
	cv::filter2D(src, prewitt2, -1, filter2);

	cv::imshow("prewitt filter x", prewitt1);
	cv::imshow("prewitt filter y", prewitt2);
	res = cv::abs(prewitt1) + cv::abs(prewitt2);



	/*int val = 0;
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			val = sqrt(prewitt1.data[i * src.cols + j] * prewitt1.data[i * src.cols + j] + prewitt2.data[i * src.cols + j] * prewitt2.data[i * src.cols + j]);
			if (val > 255) val = 255;
			if (val < 0) val = 0;
			res.data[i * src.cols + j] = val;
		}
	}*/
	cv::imshow("prewitt", res);

}
void edge_detection_sobel(const cv::Mat& src, cv::Mat& res)
{
	cv::Mat sobel1, sobel2;

	/*char data[] = { -1, 0,  1, -2, 0, 2, -1, 0, 1 };
	char data2[] = { -1, -2,  -1, 0, 0, 0, 1, 2, 1 };
	cv::Mat filter(3, 3, CV_8SC1, data);
	cv::Mat filter2(3, 3, CV_8SC1, data2);
	cv::filter2D(src, sobel1, -1, filter);
	cv::filter2D(src, sobel2, -1, filter2);

	cv::imshow("sobel filter x", sobel1);
	cv::imshow("sobel filter y", sobel2);
	res = cv::abs(sobel1) + cv::abs(sobel2);*/
	/*int val = 0;
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			val = sqrt(sobel1.data[i * src.cols + j] * sobel1.data[i * src.cols + j] + sobel2.data[i * src.cols + j] * sobel2.data[i * src.cols + j]);
			if (val > 255) val = 255;
			if (val < 0) val = 0;
			res.data[i * src.cols + j] = val;
		}
	}*/

	cv::Sobel(src, sobel1, -1, 1, 0);
	cv::Sobel(src, sobel2, -1, 0, 1);
	
	int val = 0;
	for (int i = 0; i < src.rows; ++i)
	{
		for (int j = 0; j < src.cols; ++j)
		{
			val = sqrt(sobel1.data[i * src.cols + j] * sobel1.data[i * src.cols + j] + sobel2.data[i * src.cols + j] * sobel2.data[i * src.cols + j]);
			if (val > 255) val = 255;
			if (val < 0) val = 0;
			res.data[i * src.cols + j] = val;
		}
	}
	cv::imshow("sobel filter x", sobel1);
	cv::imshow("sobel filter y", sobel2);
	cv::imshow("sobel", res);
}



cv::Mat canny_src, canny_res;
int canny_threshold1, canny_threshold2;
std::string canny_window_name;
cv::Mat gauss, gx, gy, magnitude, direction;
cv::Mat gx_double, gy_double;

void edge_detection_canny(const cv::Mat& src, cv::Mat& res, int threshold1, int threshold2)
{
	
	std::string window_name = "캐니 엣지 임계값 : " + std::to_string(threshold1) + " " + std::to_string(threshold2);
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
	canny_src = src;
	canny_res = res;
	canny_window_name = window_name;
	canny_threshold1 = threshold1;
	canny_threshold2 = threshold2;

	cv::createTrackbar("threshold1", window_name, &threshold1, 255, on_threshold1_change);
	cv::createTrackbar("threshold2", window_name, &threshold2, 255, on_threshold2_change);
	
	magnitude = cv::Mat::zeros(cv::Size(canny_src.cols, canny_src.rows), CV_64F);
	direction = cv::Mat::zeros(cv::Size(canny_src.cols, canny_src.rows), CV_64F);

	cv::GaussianBlur(canny_src, gauss, cv::Size(3, 3), 1, 1);

	cv::Sobel(gauss, gx, -1, 1, 0);
	cv::Sobel(gauss, gy, -1, 0, 1);

	cv::imshow("gx", gx);
	cv::imshow("gy", gy);

	gx.convertTo(gx_double, CV_64F, 1. / 255);
	gy.convertTo(gy_double, CV_64F, 1. / 255);

	cv::imshow("gx_double", gx_double);
	cv::imshow("gy_double", gy_double);


	double* mag = &magnitude.at<double>(0, 0);
	double* dir = &direction.at<double>(0, 0);
	double* gx_ = &gx_double.at<double>(0, 0);
	double* gy_ = &gy_double.at<double>(0, 0);
	int idx = 0;

	for (int i = 0; i < canny_src.rows; ++i)
	{
		for (int j = 0; j < canny_src.cols; ++j)
		{
			idx = i * magnitude.cols + j;
			mag[idx] = sqrt(gx_[idx] * gx_[idx] + gy_[idx] * gy_[idx]);
			dir[idx] = atan2(gy_[idx], gx_[idx]);
		}
	}

	cv::imshow("magnitude", magnitude);
	cv::imshow("direction", direction);

	/*cv::Mat gauss, gx, gy;
	cv::GaussianBlur(canny_src, gauss, cv::Size(3, 3), 1, 1);
	cv::Sobel(gauss, gx, -1, 1, 0);
	cv::Sobel(gauss, gy, -1, 0, 1);
	cv::imshow("gx", gx);
	cv::imshow("gy", gy);

	cv::Mat magnitude = cv::Mat::zeros(cv::Size(canny_src.cols, canny_src.rows), CV_64F);
	cv::Mat direction = cv::Mat::zeros(cv::Size(canny_src.cols, canny_src.rows), CV_64F);

	cv::Mat gx_double, gy_double;
	gx.convertTo(gx_double, CV_64F, 1. / 255);
	gy.convertTo(gy_double, CV_64F, 1. / 255);

	cv::imshow("gx_double", gx_double);
	cv::imshow("gy_double", gy_double);

	double* mag = &magnitude.at<double>(0, 0);
	double* dir = &direction.at<double>(0, 0);
	double* gx_ = &gx_double.at<double>(0, 0);
	double* gy_ = &gy_double.at<double>(0, 0);
	int idx = 0;
	for (int i = 0; i < canny_src.rows; ++i)
	{
		for (int j = 0; j < canny_src.cols; ++j)
		{
			idx = i * magnitude.cols + j;
			mag[idx] = sqrt(gx_[idx] * gx_[idx] + gy_[idx] * gy_[idx]);
			dir[idx] = atan2(gy_[idx], gx_[idx]);
		}
	}

	cv::imshow("magnitude", magnitude);
	cv::imshow("direction", direction);
	double threshold_low = canny_threshold1 / 255.;
	double threshold_high = canny_threshold2 / 255.;

	cv::Mat magnitude2 = cv::Mat::zeros(cv::Size(src.cols, src.rows), CV_64F);
	double* mag2 = &magnitude2.at<double>(0, 0);
	for (int i = 1; i < canny_src.rows - 1; ++i)
	{
		for (int j = 1; j < canny_src.cols - 1; ++j)
		{
			idx = i * magnitude.cols + j;
			if (mag[idx] > threshold_low)
			{
				if ( ((dir[idx] >= -22.5) && (dir[idx] < 22.5)) || (dir[idx] >= 157.5) || (dir[idx] < -157.5))
				{
					if ((mag[idx] >= mag[(i - 1) * magnitude.cols + j]) && (mag[idx] > mag[(i + 1) * magnitude.cols + j]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;
					}
				}
				else if (((dir[idx] >= 22.5) && (dir[idx] < 67.5)) || (dir[idx] >= -157.5) || (dir[idx] < -112.5))
				{
					if ((mag[idx] >= mag[(i - 1) * magnitude.cols + j - 1]) && (mag[idx] > mag[(i + 1) * magnitude.cols + j + 1]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;
					}
				}
				else if (((dir[idx] >= 67.5) && (dir[idx] < 112.5)) || (dir[idx] >= -112.5) || (dir[idx] < -67.5))
				{
					if ((mag[idx] >= mag[i * magnitude.cols + j - 1]) && (mag[idx] > mag[i * magnitude.cols + j + 1]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;

					}
				}
				else
				{
					if ((mag[idx] >= mag[(i - 1) * magnitude.cols + j + 1]) && (mag[idx] > mag[(i + 1) * magnitude.cols + j - 1]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;

					}
				}
			}
		}
	}

	for (int i = 0; i < canny_src.rows; ++i)
	{
		for (int j = 0; j < canny_src.cols; ++j)
		{
			idx = i * magnitude.cols + j;
			if (mag2[idx] == 2)
			{
				if (mag2[i * magnitude.cols + j + 1] == 1 || mag2[(i + 1) * magnitude.cols + j + 1] == 1 || mag2[(i + 1) * magnitude.cols + j] == 1 ||
					mag2[(i + 1) * magnitude.cols + j - 1] == 1 || mag2[i * magnitude.cols + j - 1] == 1 || mag2[(i - 1) * magnitude.cols + j - 1] == 1 ||
					mag2[(i - 1) * magnitude.cols + j] == 1 || mag2[(i - 1) * magnitude.cols + j + 1] == 1)
				{
					mag2[idx] = 1;
				}
				else mag2[idx] = 0;
			}
		}
	}

	cv::imshow("mag2", magnitude2);*/

	cv::imshow(canny_window_name, canny_res);
}

void on_threshold1_change(int pos, void* userdata)
{
	double* mag = &magnitude.at<double>(0, 0);
	double* dir = &direction.at<double>(0, 0);
	double* gx_ = &gx_double.at<double>(0, 0);
	double* gy_ = &gy_double.at<double>(0, 0);
	int idx = 0;
	for (int i = 0; i < canny_src.rows; ++i)
	{
		for (int j = 0; j < canny_src.cols; ++j)
		{
			idx = i * magnitude.cols + j;
			mag[idx] = sqrt(gx_[idx] * gx_[idx] + gy_[idx] * gy_[idx]);
			dir[idx] = atan2(gy_[idx], gx_[idx]);
		}
	}
	canny_threshold1 = pos;
	double threshold_low = canny_threshold1 / 255.;
	double threshold_high = canny_threshold2 / 255.;

	cv::Mat magnitude2 = cv::Mat::zeros(cv::Size(canny_src.cols, canny_src.rows), CV_64F);
	double* mag2 = &magnitude2.at<double>(0, 0);
	for (int i = 1; i < canny_src.rows - 1; ++i)
	{
		for (int j = 1; j < canny_src.cols - 1; ++j)
		{
			idx = i * magnitude.cols + j;
			if (mag[idx] > threshold_low)
			{
				if (((dir[idx] >= -22.5) && (dir[idx] < 22.5)) || (dir[idx] >= 157.5) || (dir[idx] < -157.5))
				{
					if ((mag[idx] >= mag[(i - 1) * magnitude.cols + j]) && (mag[idx] > mag[(i + 1) * magnitude.cols + j]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;
					}
				}
				else if (((dir[idx] >= 22.5) && (dir[idx] < 67.5)) || (dir[idx] >= -157.5) || (dir[idx] < -112.5))
				{
					if ((mag[idx] >= mag[(i - 1) * magnitude.cols + j - 1]) && (mag[idx] > mag[(i + 1) * magnitude.cols + j + 1]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;
					}
				}
				else if (((dir[idx] >= 67.5) && (dir[idx] < 112.5)) || (dir[idx] >= -112.5) || (dir[idx] < -67.5))
				{
					if ((mag[idx] >= mag[i * magnitude.cols + j - 1]) && (mag[idx] > mag[i * magnitude.cols + j + 1]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;

					}
				}
				else
				{
					if ((mag[idx] >= mag[(i - 1) * magnitude.cols + j + 1]) && (mag[idx] > mag[(i + 1) * magnitude.cols + j - 1]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;

					}
				}
			}
		}
	}

	for (int i = 0; i < canny_src.rows; ++i)
	{
		for (int j = 0; j < canny_src.cols; ++j)
		{
			idx = i * magnitude.cols + j;
			if (mag2[idx] == 2)
			{
				if (mag2[i * magnitude.cols + j + 1] == 1 || mag2[(i + 1) * magnitude.cols + j + 1] == 1 || mag2[(i + 1) * magnitude.cols + j] == 1 ||
					mag2[(i + 1) * magnitude.cols + j - 1] == 1 || mag2[i * magnitude.cols + j - 1] == 1 || mag2[(i - 1) * magnitude.cols + j - 1] == 1 ||
					mag2[(i - 1) * magnitude.cols + j] == 1 || mag2[(i - 1) * magnitude.cols + j + 1] == 1)
				{
					mag2[idx] = 1;
				}
				else mag2[idx] = 0;
			}
		}
	}

	cv::imshow("my_canny", magnitude2);
	
	canny_threshold1 = pos;
	cv::Mat sobel_x, sobel_y;
	cv::Sobel(canny_src, sobel_x, -1, 1, 0);
	cv::Sobel(canny_src, sobel_y, -1, 0, 1);
	sobel_x.convertTo(sobel_x, CV_16SC1);
	sobel_y.convertTo(sobel_y, CV_16SC1);
	cv::Canny(sobel_x, sobel_y, canny_res, canny_threshold1, canny_threshold2);
	cv::imshow(canny_window_name, canny_res);
}
void on_threshold2_change(int pos, void* userdata)
{
	double* mag = &magnitude.at<double>(0, 0);
	double* dir = &direction.at<double>(0, 0);
	double* gx_ = &gx_double.at<double>(0, 0);
	double* gy_ = &gy_double.at<double>(0, 0);
	int idx = 0;
	for (int i = 0; i < canny_src.rows; ++i)
	{
		for (int j = 0; j < canny_src.cols; ++j)
		{
			idx = i * magnitude.cols + j;
			mag[idx] = sqrt(gx_[idx] * gx_[idx] + gy_[idx] * gy_[idx]);
			dir[idx] = atan2(gy_[idx], gx_[idx]);
		}
	}
	canny_threshold2 = pos;
	double threshold_low = canny_threshold1 / 255.;
	double threshold_high = canny_threshold2 / 255.;

	cv::Mat magnitude2 = cv::Mat::zeros(cv::Size(canny_src.cols, canny_src.rows), CV_64F);
	double* mag2 = &magnitude2.at<double>(0, 0);
	for (int i = 1; i < canny_src.rows - 1; ++i)
	{
		for (int j = 1; j < canny_src.cols - 1; ++j)
		{
			idx = i * magnitude.cols + j;
			if (mag[idx] > threshold_low)
			{
				if (((dir[idx] >= -22.5) && (dir[idx] < 22.5)) || (dir[idx] >= 157.5) || (dir[idx] < -157.5))
				{
					if ((mag[idx] >= mag[(i - 1) * magnitude.cols + j]) && (mag[idx] > mag[(i + 1) * magnitude.cols + j]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;
					}
				}
				else if (((dir[idx] >= 22.5) && (dir[idx] < 67.5)) || (dir[idx] >= -157.5) || (dir[idx] < -112.5))
				{
					if ((mag[idx] >= mag[(i - 1) * magnitude.cols + j - 1]) && (mag[idx] > mag[(i + 1) * magnitude.cols + j + 1]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;
					}
				}
				else if (((dir[idx] >= 67.5) && (dir[idx] < 112.5)) || (dir[idx] >= -112.5) || (dir[idx] < -67.5))
				{
					if ((mag[idx] >= mag[i * magnitude.cols + j - 1]) && (mag[idx] > mag[i * magnitude.cols + j + 1]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;

					}
				}
				else
				{
					if ((mag[idx] >= mag[(i - 1) * magnitude.cols + j + 1]) && (mag[idx] > mag[(i + 1) * magnitude.cols + j - 1]))
					{
						if (mag[idx] > threshold_high)
							mag2[idx] = 1;
						else mag2[idx] = 2;

					}
				}
			}
		}
	}

	for (int i = 0; i < canny_src.rows; ++i)
	{
		for (int j = 0; j < canny_src.cols; ++j)
		{
			idx = i * magnitude.cols + j;
			if (mag2[idx] == 2)
			{
				if (mag2[i * magnitude.cols + j + 1] == 1 || mag2[(i + 1) * magnitude.cols + j + 1] == 1 || mag2[(i + 1) * magnitude.cols + j] == 1 ||
					mag2[(i + 1) * magnitude.cols + j - 1] == 1 || mag2[i * magnitude.cols + j - 1] == 1 || mag2[(i - 1) * magnitude.cols + j - 1] == 1 ||
					mag2[(i - 1) * magnitude.cols + j] == 1 || mag2[(i - 1) * magnitude.cols + j + 1] == 1)
				{
					mag2[idx] = 1;
				}
				else mag2[idx] = 0;
			}
		}
	}

	cv::imshow("my canny", magnitude2);

	canny_threshold2 = pos;
	cv::Mat sobel_x, sobel_y;
	cv::Sobel(canny_src, sobel_x, -1, 1, 0);
	cv::Sobel(canny_src, sobel_y, -1, 0, 1);
	sobel_x.convertTo(sobel_x, CV_16SC1);
	sobel_y.convertTo(sobel_y, CV_16SC1);
	cv::Canny(sobel_x, sobel_y, canny_res, canny_threshold1, pos);
	cv::imshow(canny_window_name, canny_res);

}

