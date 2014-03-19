#include <iostream>
#include <opencv2/opencv.hpp>

#include "../../saliency/SpectralResidualSaliency.hpp"

int main(int argc, char **argv) {
  if (argc < 2) {
    cout<<"Please input an image filename"<<endl;
    return 1;
  }

  Mat_<Vec3b> inputImage = imread(argv[1]);
  Mat_<float> grayImage;

  cvtColor(inputImage, grayImage, CV_BGR2GRAY);
  Mat_<float> saliencyMap;

  spectralResidualSaliencyMap(grayImage, saliencyMap, 3, 1, 400);
  double min, max;

  minMaxLoc(saliencyMap, &min, &max);

  cout<<"min = "<<min<<", max = "<<max<<endl;

  imshow("source", inputImage);
  imshow("gray", grayImage);
  imshow("saliency", (saliencyMap - min) / max);
  waitKey(0);

  return 0;
}
