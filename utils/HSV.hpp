#ifndef __HSV_HPP__
#define __HSV_HPP__

class HSV{
  public:
    ~HSV();
    HSV(int size);

  public:
    float*H;
    float*S;
    float*V;
    int size;
};

#endif