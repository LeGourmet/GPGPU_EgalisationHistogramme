#ifndef __CODE_GPU_HPP__
#define __CODE_GPU_HPP__

#include "./utils/HSV.hpp"
#include "./utils/image.hpp"

__global__ void computeRGB2HSV( const unsigned char* data, const int sizeData, float* const hsv);
float rgb2hsv_GPU(Image* img, HSV* hsv);

__global__ void computeRGB2HSV( const float* data, const int sizeData, unsigned char* const hsv);
float hsv2rgb_GPU(HSV* hsv, Image* img);

__global__ void computeHisto( const float* data, const int sizeData, int* const histo);
float histogram_GPU(HSV *hsv, int* histo);

__global__ void computeRepart( const int* histo, int* const repart );
float repart_GPU(int* histo, int* repart);

__global__ void computeEqualization( const int *repart, const int sizeV, const float* V_in, float* const V_out);
float equalization_GPU(HSV *hsv, int* repart);

#endif