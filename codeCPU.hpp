#ifndef __CODE_CPU_HPP__
#define __CODE_CPU_HPP__

#include "./utils/image.hpp"
#include "./utils/HSV.hpp"

float rgb2hsv_CPU(Image *img, HSV *hsv);
float hsv2rgb_CPU(HSV *hsv, Image *img);
float histogram_CPU(HSV *hsv, int *histo);
float repart_CPU(int *histo, int *repart);
float equalization_CPU(HSV *hsv, int *repart);

#endif