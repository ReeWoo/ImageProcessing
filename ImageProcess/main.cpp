#include <stdio.h>
#include <chrono>
#include "image_utils.h"

int main()
{
	printf("*** project start ***\n");
	cv::Mat img, result, color_img, template_img;
	image_read("C:\\Users\\yoonw\\Downloads\\lena.jpg", color_img, cv::IMREAD_COLOR);
	//image_read("C:\\Users\\yoonw\\Downloads\\lena.jpg", img, cv::IMREAD_GRAYSCALE);
	image_read("C:\\Users\\yoonw\\Downloads\\lenna_eye.png", template_img, cv::IMREAD_GRAYSCALE);
	
	//image_read("C:\\Users\\yoonw\\Downloads\\TEST1.png", img, cv::IMREAD_GRAYSCALE);
	//image_read("C:\\Users\\yoonw\\Downloads\\TEST1.png", color_img, cv::IMREAD_COLOR);
	
	//brightness_inverse(img, result);
	//add_scalar(img, 150, result);
	//multiply_scalar(img, 1.5, result);
	//draw_histogram(img, "original");
	///*improve_contrast(img, 1.5, result);
	//draw_histogram(result);
	//gamma_correction(img, 0.6, result);
	//draw_histogram(result);*/
	//histogram_stretching(img, result);
	//draw_histogram(result, "hs");
	//histogram_equalization(img, result);
	//draw_histogram(result, "he");

	//cv::Mat img2 = cv::Mat::ones(cv::Size(img.cols, img.rows), CV_8UC1);
	//img2 = img2 * 255;
	//int width = img.cols / 2;
	//int height = img.rows / 2;
	////cv::rectangle(img2, cv::Rect(width - width / 2, height - height / 2, width, height), cv::Scalar(0), -1);
	//cv::circle(img2, cv::Point(width, height), width / 2, cv::Scalar(0), -1);

	//cv::imshow("img2", img2);
	//add_image(img, img2, result);
	////sub_image(img, img2, result);
	//sub_image(img2, img, result);

	//cv::Mat noise_image(img.size(), CV_8UC1);
	//double average = 0.0;
	//double std = 30.0;
	//randn(noise_image, cv::Scalar::all(average), cv:: Scalar::all(std));
	//
	//cv::Mat noise_image2(img.size(), CV_8UC1);
	//average = 5.0;
	//std = 20.0;
	//randn(noise_image2, cv::Scalar::all(average), cv::Scalar::all(std));

	//cv::Mat noise_image3(img.size(), CV_8UC1);
	//average = 10.0;
	//std = 25.0;
	//randn(noise_image3, cv::Scalar::all(average), cv::Scalar::all(std));

	//cv::Mat noise_image4(img.size(), CV_8UC1);
	//average = 15.0;
	//std = 15.0;
	//randn(noise_image4, cv::Scalar::all(average), cv::Scalar::all(std));

	//cv::Mat noise_image5(img.size(), CV_8UC1);
	//average = 20.0;
	//std = 10.0;
	//randn(noise_image5, cv::Scalar::all(average), cv::Scalar::all(std));
	//addWeighted(img, 1.0, noise_image, 1.0, 0.0, noise_image);
	//addWeighted(img, 1.0, noise_image2, 1.0, 0.0, noise_image2);
	//addWeighted(img, 1.0, noise_image3, 1.0, 0.0, noise_image3);
	//addWeighted(img, 1.0, noise_image4, 1.0, 0.0, noise_image4);
	//addWeighted(img, 1.0, noise_image5, 1.0, 0.0, noise_image5);
	///*noise_image += img;
	//noise_image2 += img;*/
	//cv::imshow("noise_image", noise_image);
	//cv::imshow("noise_image2", noise_image2);
	//cv::imshow("noise_image3", noise_image3);
	//cv::imshow("noise_image4", noise_image4);
	//cv::imshow("noise_image5", noise_image5);

	//std::vector<cv::Mat> vec;
	//vec.push_back(noise_image);
	//vec.push_back(noise_image2);
	//vec.push_back(noise_image3);
	//vec.push_back(noise_image4);
	//vec.push_back(noise_image5);
	//avg_image(vec, result);

	///*cv::Mat img1, img3;
	//image_read("C:\\Users\\yoonw\\Downloads\\img1.png", img1, cv::IMREAD_GRAYSCALE);
	//image_read("C:\\Users\\yoonw\\Downloads\\img2.png", img3, cv::IMREAD_GRAYSCALE);

	//cv::imshow("image1", img1);
	//cv::imshow("image2", img3);
	//abs_sub_image(img1, img3, result);*/
	//
	//img2 = cv::Mat::ones(cv::Size(img.rows, img.cols), CV_8UC1);
	//img2 *= 127;
	//and_image(img, img2, result);
	//or_image(img, img2, result);

	//for (int i = 0; i < 8; ++i)
	//	bit_plane(img, result, i);

	//average_filter(img, result, 3);
	//average_filter(img, result, 5);
	//weighted_filter(img, result);
	//double sigma = 1;
	//gaussian(img, result, 2 * 4 * sigma + 1, sigma, sigma);
	//sigma = 4;
	//gaussian(img, result, 2 * 4 * sigma + 1, sigma, sigma);
	//sharpening(img, result);
	//laplacian(img, result);
	//high_boost(img, result, 1.0);
	//high_boost(img, result, 1.5);
	//add_gaussian_noise(img, result, 0, 1, 5);
	//salt_pepper(img, result, 5);
	//median_filter(img, result);
	//translation(img, result, -100, -50);
	//scale_transform(img, result, 3.3, 3.4);
	//scale_transform_with_bilinear(img, result, 3.3 , 3.4);
	//scale_transform_3order_interpolation(img, result, 3.3, 3.4);
	//edge_detection_roberts(img, result);
	//edge_detection_prewitt(img, result);
	////edge_detection_sobel(img, result);
	////edge_detection_canny(img, result, 30, 60);
	//image_read("C:\\Users\\yoonw\\Downloads\\car_lane.png", img, cv::IMREAD_GRAYSCALE);
	//image_read("C:\\Users\\yoonw\\Downloads\\test_harris.jpg", img, cv::IMREAD_GRAYSCALE);
	//
	//image_read("C:\\Users\\yoonw\\Downloads\\test_harris.jpg", img2, cv::IMREAD_COLOR);
	//cv::resize(img, img, cv::Size(300, 300));
	//cv::resize(img2, img2, cv::Size(300, 300));
	////hough_transform(img, result);
 //   std::vector<cv::Point2f> vector =  harris_corner_detection(img, result);
	//std::cout << "vec : " << vector.size() << std::endl;
	//for (int i = 0; i < vector.size(); ++i)
	//{
	//	cv::circle(img2, vector[i], 5, cv::Scalar(0, 0, 255), 2);
	//}
	//cv::imshow("harris corner", img2);
	
	/*cv::Mat gray_img, gray_sobel;
	color_inverse(img, result);
	color2gray(img, gray_img);
	color_edge_detection(img, result);
	edge_detection_prewitt(gray_img, gray_sobel);
	color_histogram_equalization(img, result);*/

	/*cv::resize(img, img, cv::Size(50, 50));
	cv::resize(color_img, color_img, cv::Size(50, 50));*/

	/*cv::Mat binary_img;
	binarization(img, binary_img, 80);
	contour_with_erosion(binary_img, result);*/
	/*erosion(binary_img, result);
	dilation(binary_img, result);

	opening(binary_img, result);
	closing(binary_img, result);*/
	
	//iterative_binarization(img, result);
	//std::vector<int> table;
	//image_labeling(result, color_img, table);
	//contour_trace(result, color_img);
	
	cv::cvtColor(color_img, img, cv::COLOR_BGR2GRAY);
	matching_template(img, template_img, color_img);
	cv::waitKey(0); 

	return 0;
}