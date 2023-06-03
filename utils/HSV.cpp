#include <cstdlib>
#include "HSV.hpp"

HSV::~HSV(){
	delete[] H;
	delete[] S;
	delete[] V;
}

HSV::HSV(int size){
	this->size = size;
	this->H = new float[size];
	this->S = new float[size];
	this->V = new float[size];
}
