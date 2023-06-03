#include <math.h>
#ifdef _WIN32
	#define NOMINMAX
#endif
#include "codeCPU.hpp"
#include "utils/chrono/chronoCPU.hpp"

// ********************************************* RGB->HSV *********************************************
float rgb2hsv_CPU(Image *img, HSV *hsv){
	ChronoCPU chr;
	chr.start();
	
	float max, min, delta, r, g, b;
		
	for(int i=0; i<img->_width*img->_height ;i++){
		r =	img->_pixels[i*3]/255.f;
		g = img->_pixels[i*3+1]/255.f;
		b = img->_pixels[i*3+2]/255.f;

		max = std::max(r,std::max(g,b));
		min = std::min(r,std::min(g,b));
		delta = max-min;

		hsv->H[i] = (min==max ? 0.f :
				  	(max==r ? fmod((60*((g-b)/delta))+360,360) :
				  	(max==g ? 60*((b-r)/delta)+120 : 
				   	          60*((r-g)/delta)+240 )));

		hsv->S[i] = (max==0.f ? 0.f : delta/max);
		hsv->V[i] = max;
	}

	chr.stop();
	return chr.elapsedTime();
}

// ********************************************* HSV->RGB *********************************************
float hsv2rgb_CPU(HSV *hsv, Image *img){
	ChronoCPU chr;
	chr.start();

	for(int i=0; i<img->_width*img->_height ;i++){

		float t = (hsv->H[i])/60.f;
		float r = (fmod(t,2)-1);
		float c = hsv->V[i]*hsv->S[i];
		float x = c*(1.f-(r>0 ? r : -1*r));
		float m = hsv->V[i]-c;

		float* res = 
				(t<1.f ? new float[3]{c,x,0} :
				(t<2.f ? new float[3]{x,c,0} :
				(t<3.f ? new float[3]{0,c,x} :
				(t<4.f ? new float[3]{0,x,c} :
				(t<5.f ? new float[3]{x,0,c} : 
						 new float[3]{c,0,x} )))));

		img->_pixels[i*3] = (res[0]+m)*255;
		img->_pixels[i*3+1] = (res[1]+m)*255;
		img->_pixels[i*3+2] = (res[2]+m)*255;
	}

	chr.stop();
	return chr.elapsedTime();
}

// ********************************************* HISTOGRAM *********************************************
float histogram_CPU(HSV *hsv, int *histo){
	ChronoCPU chr;
	chr.start();
	
	for(int i=0; i<hsv->size ;i++){
		histo[(int)(hsv->V[i]*255.f)] +=1;	
	}

	chr.stop();
	return chr.elapsedTime();
}

// ********************************************* REPARTITION *********************************************
float repart_CPU(int *histo, int *repart){
	ChronoCPU chr;
	chr.start();

	int res = 0;
	for(int i=0; i<256 ;i++){
		res += histo[i];
		repart[i] = res;
	}

	chr.stop();
	return chr.elapsedTime();
}

// ********************************************* EQUALIZATION *********************************************
float equalization_CPU(HSV *hsv, int *repart){
	ChronoCPU chr;
	chr.start();

	for(int i=0; i<hsv->size ;i++){
		hsv->V[i] = ((255.f/hsv->size)*repart[((int)(hsv->V[i]*255.f))])/255.f;
	}

	chr.stop();
	return chr.elapsedTime();
}
