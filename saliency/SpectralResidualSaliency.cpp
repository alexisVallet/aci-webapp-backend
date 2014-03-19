#include "SpectralResidualSaliency.hpp"

void spectralResidualSaliencyMap(
  const Mat_<float> &image, 
  Mat_<float> &saliencyMap, 
  int avgFilterSize, 
  float gaussianSigma,
  int downSamplingMaxDim) {

  // down sampling of the image to a size smaller than 64x64
  Size newSize = image.cols > image.rows ? 
    Size(downSamplingMaxDim, image.rows * downSamplingMaxDim / image.cols) :
    Size(image.cols * downSamplingMaxDim / image.rows, downSamplingMaxDim);
  Mat downSampled8b;

  resize(image, downSampled8b, newSize);
  Mat downSampled = Mat_<float>(downSampled8b) / 255.;

  vector<Mat> planes;
  planes.push_back(downSampled);
  planes.push_back(Mat::zeros(newSize, CV_32FC1));
  Mat downSampledComplexDyn(newSize, CV_32FC2);
  merge(planes, downSampledComplexDyn);
  Mat_<Vec2f> downSampledComplex = Mat_<Vec2f>(downSampledComplexDyn);

  // compute its residual
  Mat_<Vec2f> fourierSpectrum;

  cout<<"computing residual"<<endl;
  dft(downSampledComplex, fourierSpectrum, DFT_COMPLEX_OUTPUT);
  Mat_<float> parts[2];

  split(fourierSpectrum, parts);

  Mat_<float> logSpectrum;
  log(parts[0], logSpectrum);
  Mat_<float> filteredSpectrum;
  Mat_<float> avgFilter = 
    Mat_<float>::ones(avgFilterSize, avgFilterSize) / avgFilterSize * avgFilterSize;
  
  filter2D(logSpectrum, filteredSpectrum, -1, avgFilter);
  Mat_<float> residual = logSpectrum - filteredSpectrum;

  // put back the residual with the unchanged imaginary part
  Mat_<Vec2f> residualSpectrum(residual.rows, residual.cols);

  for (int i = 0; i < residual.rows; i++) {
    for (int j = 0; j < residual.cols; j++) {
      residualSpectrum(i,j) = Vec2f(residual(i,j), parts[1](i,j));
    }
  }

  // turn the spectrum back to an image, with some filtering
  Mat_<Vec2f> exponentiated(residualSpectrum.rows, residualSpectrum.cols);
  Mat_<Vec2f> unfilteredOutput;
  Mat_<float> squared;

  // exponentiating the coefficients manually, because opencv sucks at complex arithmetic
  for (int i = 0; i < residualSpectrum.rows; i++) {
    for (int j = 0; j < residualSpectrum.cols; j++) {
      complex<float> coeff(residualSpectrum(i,j)[0], residualSpectrum(i,j)[1]);
      complex<float> expCoeff = exp(coeff);
      
      exponentiated(i,j)[0] = expCoeff.real();
      exponentiated(i,j)[1] = expCoeff.imag();
    }
  }

  cout<<"running inverse dft"<<endl;
  dft(exponentiated, unfilteredOutput, DFT_INVERSE);
  Mat_<float> unfilteredParts[2];
  split(unfilteredOutput, unfilteredParts);
  pow(unfilteredParts[0], 2, squared);
  GaussianBlur(squared, saliencyMap, Size(0,0), gaussianSigma);
}

