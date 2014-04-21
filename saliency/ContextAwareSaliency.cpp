#include "ContextAwareSaliency.hpp"

void contextAwareSaliency(const Mat_<Vec3b> &image, Mat_<float> &saliency, vector<int> &scales, vector<float> &multiScales, int k, int maxDimension) {
  // First resize the input image
  Size newSize =
    image.rows > image.cols ?
    Size(image.cols * maxDimension / image.rows, maxDimension) :
    Size(maxDimension, image.rows * maxDimension / image.cols);
  Mat_<Vec3b> resizedImage;

  resize(image, resizedImage, newSize);

  
}
