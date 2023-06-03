#include <iostream>
#include "./utils/image.hpp"
#include "./utils/HSV.hpp"
#include "codeCPU.hpp"
#include "codeGPU.hpp"

// nvcc -o projet main.cu ./utils/image.cpp codeCPU.cpp codeGPU.cu ./utils/HSV.cpp ./utils/chrono/chronoCPU.cpp ./utils/chrono/chronoGPU.cu -O1
// nvcc -o projet main.cu ./utils/image.cpp codeCPU.cpp codeGPU.cu ./utils/HSV.cpp ./utils/chrono/chronoCPU.cpp ./utils/chrono/chronoGPU.cu -O1 -arch=compute_50

int main( int argc, char **argv ) {
	Image img = Image();
	img.load("./images/Chateau.png");
	int size = img._height*img._width;

	std::cout << "============================================"	<< std::endl;
	std::cout << "         Sequential version on CPU          "	<< std::endl;
	std::cout << "============================================"	<< std::endl;
	
	Image cpuImg = Image();
	cpuImg._height = img._height;
	cpuImg._width = img._width;
	cpuImg._nbChannels = 3;
	cpuImg._pixels = new unsigned char[3*size];

	HSV hsv_cpu = HSV(size);
	std::cout << "conversion rgb->hsv :\t\t" << rgb2hsv_CPU(&img,&hsv_cpu) << std::endl;

	int*histo_cpu = new int[256];
	int*repart_cpu = new int[256];
	for(int i=0; i<256 ;i++){histo_cpu[i]=0;repart_cpu[i]=0;}

	std::cout << "remplissage de l'histogramme :\t" << histogram_CPU(&hsv_cpu,histo_cpu) << std::endl;
	std::cout << "répartition de l'histogramme :\t" << repart_CPU(histo_cpu,repart_cpu) << std::endl;
	std::cout << "égalisation de l'histogramme :\t" << equalization_CPU(&hsv_cpu,repart_cpu) << std::endl;

	std::cout << "conversion hsv->rgb :\t\t" << hsv2rgb_CPU(&hsv_cpu,&cpuImg) << std::endl << std::endl;
	cpuImg.save("./resultats/cpuImg.png");

	std::cout << "============================================"	<< std::endl;
	std::cout << "         Parallel version on GPU            "	<< std::endl;
	std::cout << "============================================"	<< std::endl;

	Image gpuImg = Image();
	gpuImg._height = img._height;
	gpuImg._width = img._width;
	gpuImg._nbChannels = 3;
	gpuImg._pixels = new unsigned char[3*size];

	HSV hsv_gpu = HSV(size);
	std::cout << "conversion rgb->hsv :\t\t" << rgb2hsv_GPU(&img,&hsv_gpu) << std::endl;

	int*histo_gpu = new int[256];
	int*repart_gpu = new int[256];
	for(int i=0; i<256 ;i++){histo_gpu[i]=0;repart_gpu[i]=0;}

	std::cout << "remplissage de l'histogramme :\t" << histogram_GPU(&hsv_gpu,histo_gpu) << std::endl;
	std::cout << "répartition de l'histogramme :\t" << repart_GPU(histo_gpu,repart_gpu) << std::endl;
	std::cout << "égalisation de l'histogramme :\t" << equalization_GPU(&hsv_gpu,repart_gpu) << std::endl;

	std::cout << "conversion hsv->rgb :\t\t" << hsv2rgb_GPU(&hsv_gpu,&gpuImg) << std::endl << std::endl;
	gpuImg.save("./resultats/gpuImg.png");
	
	return EXIT_SUCCESS;
}