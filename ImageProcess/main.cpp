#include <stdio.h>
#include "image_utils.h"

int main()
{
	printf("*** project start ***\n");
	cv::Mat img;
	image_read("C:\\Users\\yoonw\\Downloads\\lena.jpg", img, cv::IMREAD_GRAYSCALE);
	return 0;
}