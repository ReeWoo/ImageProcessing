#include <stdio.h>
#include "image_utils.h"

int main()
{
	printf("*** project start ***\n");
	cv::Mat img, result;
	image_read("C:\\Users\\yoonw\\Downloads\\lena.jpg", img, cv::IMREAD_GRAYSCALE);
	brightness_inverse(img, result);
	add_scalar(img, -100, result);
	multiply_scalar(img, 1.5, result);
	cv::waitKey(0);
	return 0;
}