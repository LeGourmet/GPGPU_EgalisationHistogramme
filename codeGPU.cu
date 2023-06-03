#include "codeGPU.hpp"
#include "utils/chrono/common.hpp"
#include "utils/chrono/chronoGPU.hpp"


// ********************************************* RGB2HSV *********************************************
__global__ void computeRGB2HSV(const unsigned char* data, const int sizeData, float* const hsv) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	float cmax, cmin, delta, r, g, b, h;
	while(i<sizeData){
		
		r = data[i*3] / 255.0f;
		g = data[i*3+1] / 255.0f;
		b = data[i*3+2] / 255.0f;
		
		cmax = fmax(r, fmax(g, b));
		cmin = fmin(r, fmin(g, b));
		delta = cmax - cmin;
		
		hsv[i*3+2] = cmax;
		
		if(cmax == 0.0f) {
			h = hsv[i*3+1] = 0.0f;
		} else {
			hsv[i*3+1] = delta / cmax;
			if(delta < 0.001f) {
				h = 0.0f;
			} else {
				if(cmax == r) {
					h = 60.0f * (g - b)/delta;
					if(h < 0.0f) { h += 360.0f; }
				} else if(cmax == g) {
					h = 60.0f * (2 + (b - r)/delta);
				} else {
					h = 60.0f * (4 + (r - g)/delta);
				}
			}		
		}
		hsv[i*3] = h;
		i += blockDim.x * gridDim.x;
	}
}

float rgb2hsv_GPU(Image* img, HSV* hsv){
	// data device
	unsigned char* dev_inPtr;
	float* dev_outPtr;

	float* resultArr = new float[hsv->size*3];

	unsigned long sData = hsv->size*3*sizeof(unsigned char);
	unsigned long sRes = hsv->size*3*sizeof(float);

	ChronoGPU chr;
	chr.start();

	// Allocate memory on Device
		HANDLE_ERROR(cudaMalloc(&dev_inPtr, sData));
		HANDLE_ERROR(cudaMalloc(&dev_outPtr, sRes));

	// Copy from Host to Device
		HANDLE_ERROR(cudaMemcpy(dev_inPtr, img->_pixels, sData, cudaMemcpyHostToDevice));

	// Launch kernel
		computeRGB2HSV <<< 1024, 256 >>>(dev_inPtr,hsv->size,dev_outPtr);

	// Copy from Device to Host
		HANDLE_ERROR(cudaMemcpy(resultArr, dev_outPtr, sRes, cudaMemcpyDeviceToHost));

	// Free memory on Device
		HANDLE_ERROR(cudaFree(dev_outPtr));
		HANDLE_ERROR(cudaFree(dev_inPtr));

	// Delinearize data
	for(int i=0;i<hsv->size;i++){
		hsv->H[i] = resultArr[i*3];
		hsv->S[i] = resultArr[i*3+1];
		hsv->V[i] = resultArr[i*3+2];
	}

	chr.stop();

	return chr.elapsedTime();
}

// ********************************************* HSV2RGB *********************************************
__global__ void computeHSV2RGB(const float* data, const int sizeData, unsigned char* const pixels) {
	//basé sur https://docs.nvidia.com/cuda/npp/group__hsvtorgb.html
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	float h, s, v, h2, frac, m, n, k, r, b, g;
	while(i<sizeData){
		h = data[i*3];
		s = data[i*3+1];
		v = data[i*3+2];

		h2 = h/60.f; 
		frac = h2 - floorf(h2);
		m = v * (1.0f - s);  			 
		n = v * (1.0f - s * frac);	 
		k = v * (1.0f - s * (1.0f - frac));
		
		if (h2>=0 && h2<1.f) {
			r = v;g = k;b = m;
		} else if (h2<2.f) {
			r = n;g = v;b = m;
		} else if (h2<3.f) {
			r = m;g = v;b = k;
		} else if (h2<4.f) {
			r = m;g = n;b = v;
		} else if (h2<5.f) {
			r = k;g = m;b = v;
		} else if (h2<6.f){
			r = v;g = m;b = n;
		}
		
		pixels[i*3] = r*255.f;
		pixels[i*3+1] = g*255.f;
		pixels[i*3+2] = b*255.f;
		i += blockDim.x * gridDim.x;
	}
}

float hsv2rgb_GPU(HSV* hsv, Image* img){
	// data device
	float* dev_inPtr;
	unsigned char* dev_outPtr;

	unsigned long sData = hsv->size*3*sizeof(float);
	unsigned long sRes = hsv->size*3*sizeof(unsigned char);

	// linearize data
	float* data = new float[hsv->size*3];
	for(int i = 0;i<hsv->size;i++){
		data[i*3] = hsv->H[i];
		data[i*3+1] = hsv->S[i];
		data[i*3+2] = hsv->V[i];
	}

	ChronoGPU chr;
	chr.start();

	// Allocate memory on Device
		HANDLE_ERROR(cudaMalloc(&dev_inPtr, sData));
		HANDLE_ERROR(cudaMalloc(&dev_outPtr, sRes));

	// Copy from Host to Device
		HANDLE_ERROR(cudaMemcpy(dev_inPtr, data, sData, cudaMemcpyHostToDevice));

	// Launch kernel
		computeHSV2RGB <<< 512, 1024 >>>(dev_inPtr,hsv->size,dev_outPtr);
	
	// Copy from Device to Host
		HANDLE_ERROR(cudaMemcpy(img->_pixels, dev_outPtr, sRes, cudaMemcpyDeviceToHost));

	// Free memory on Device
		HANDLE_ERROR(cudaFree(dev_outPtr));
		HANDLE_ERROR(cudaFree(dev_inPtr));

	chr.stop();


	return chr.elapsedTime();
}

// ********************************************* HISTO *********************************************
__global__ void computeHisto(const float* data, const int sizeData, int* const histo) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	while(i<sizeData){
		int j = data[i]*255.f;
		atomicAdd(&histo[j],1);
		i += blockDim.x * gridDim.x;
	}
}

float histogram_GPU(HSV* hsv, int* histo) {
	// data device
	float* dev_inPtr;
	int* dev_outPtr;
	
	unsigned long sData = hsv->size*sizeof(float);
	unsigned long sHisto =  256*sizeof(int);

	ChronoGPU chr;
	chr.start();

	// Allocate memory on Device
		HANDLE_ERROR(cudaMalloc(&dev_inPtr, sData));
		HANDLE_ERROR(cudaMalloc(&dev_outPtr, sHisto));

	// Copy from Host to Device
		HANDLE_ERROR(cudaMemcpy(dev_inPtr, hsv->V, sData, cudaMemcpyHostToDevice));

	// Launch kernel
		computeHisto <<< 512, 1024 >>>(dev_inPtr,hsv->size,dev_outPtr);

	// Copy from Device to Host
		HANDLE_ERROR(cudaMemcpy(histo, dev_outPtr, sHisto, cudaMemcpyDeviceToHost));

	// Free memory on Device
		HANDLE_ERROR(cudaFree(dev_outPtr));
		HANDLE_ERROR(cudaFree(dev_inPtr));

	chr.stop();

	return chr.elapsedTime();
}

// ********************************************* REPARTITION *********************************************
__global__ void computeRepart( const int* histo, int* const repart ){
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	int sum = 0;
	int nbElem = 2;

	while( i < 256 ){
		repart[i]=histo[i];
		__syncthreads();
		while(nbElem<=256){
			if(i*nbElem>256){break;}
			sum = repart[i*nbElem+(nbElem/2)-1];
			for(int j=nbElem/2; j<nbElem ;j++){
				repart[i*nbElem+j] += sum;
			}
			nbElem *= 2;
		}

		i += blockDim.x * gridDim.x;;
	}
}

float repart_GPU(int* histo, int* repart){
	// data device
	int* dev_inPtr;
	int* dev_outPtr;
	
	unsigned long size = 256*sizeof(int);

	ChronoGPU chr;
	chr.start();

	// Allocate memory on Device
		HANDLE_ERROR(cudaMalloc(&dev_inPtr, size));
		HANDLE_ERROR(cudaMalloc(&dev_outPtr, size));

	// Copy from Host to Device
		HANDLE_ERROR(cudaMemcpy(dev_inPtr, histo, size, cudaMemcpyHostToDevice));

	// Launch kernel
		computeRepart <<< 512, 1024 >>>(dev_inPtr,dev_outPtr);

	// Copy from Device to Host
		HANDLE_ERROR(cudaMemcpy(repart, dev_outPtr, size, cudaMemcpyDeviceToHost));

	// Free memory on Device
		HANDLE_ERROR(cudaFree(dev_outPtr));
		HANDLE_ERROR(cudaFree(dev_inPtr));

		chr.stop();

	return chr.elapsedTime();
}

// ********************************************* EQUALISATION *********************************************
__global__ void computeEqualization( const int *repart, const int sizeV, const float* V_in  ,float* const V_out){
 	int i = threadIdx.x + blockIdx.x*blockDim.x;
	while(i<sizeV){
		V_out[i] = ((255.f/(sizeV))*repart[(int)(V_in[i]*255.f)])/255.f;
		i += blockDim.x * gridDim.x;
	}
}

float equalization_GPU(HSV *hsv, int* repart){
	// data device
	int* repart_inPtr;
	float* V_inPtr;
	float* V_outPtr;
	
	unsigned long sRepart = 256*sizeof(int);
	unsigned long sV =      hsv->size*sizeof(float);

	ChronoGPU chr;
	chr.start();
	// Allocate memory on Device
		HANDLE_ERROR(cudaMalloc(&repart_inPtr, sRepart));
		HANDLE_ERROR(cudaMalloc(&V_inPtr, sV));
		HANDLE_ERROR(cudaMalloc(&V_outPtr, sV));

	// Copy from Host to Device
		HANDLE_ERROR(cudaMemcpy(repart_inPtr, repart, sRepart, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(V_inPtr, hsv->V, sV, cudaMemcpyHostToDevice));

	// Launch kernel
		computeEqualization <<< 128, 1024 >>>(repart_inPtr,hsv->size,V_inPtr,V_outPtr);

	// Copy from Device to Host
		HANDLE_ERROR(cudaMemcpy(hsv->V, V_outPtr, sV, cudaMemcpyDeviceToHost));

	// Free memory on Device
		HANDLE_ERROR(cudaFree(V_outPtr));
		HANDLE_ERROR(cudaFree(V_inPtr));
		HANDLE_ERROR(cudaFree(repart_inPtr));

		chr.stop();

	return chr.elapsedTime();
}