#include "SpectralResidualSaliency.hpp"

static void showScaled(string windowName, Mat &image) {
  double min, max;

  minMaxLoc(image, &min, &max);
  cout<<windowName<<" min = "<<min<<", max = "<<max<<endl;
  imshow(windowName, (image - min) / (max - min));
  waitKey(0);
}

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

  imshow("downSampled", downSampled);

  vector<Mat> planes;
  planes.push_back(downSampled);
  planes.push_back(Mat::zeros(newSize, CV_32FC1));
  Mat downSampledComplex(newSize, CV_32FC2);
  merge(planes, downSampledComplex);

  // compute its residual
  // first compute its spectrum
  Mat fourierSpectrum;

  cout<<"computing residual"<<endl;
  dft(downSampledComplex, fourierSpectrum, DFT_COMPLEX_OUTPUT);
  Mat parts[2];

  split(fourierSpectrum, parts);

  // apply log scaling
  // first normalize the range of values to nonnegative values
  double spectrumMin;
  minMaxLoc(parts[0], &spectrumMin);
  parts[0] = parts[0] - spectrumMin + 1;
  Mat logSpectrum;
  log(parts[0], logSpectrum);
  showScaled("spectrum", parts[0]);
  showScaled("logSpectrum", logSpectrum);

  Mat logSpectrumParts[] = {logSpectrum, parts[1]};
  Mat fullLogSpectrum, invDFTLog;
  merge(logSpectrumParts, 2, fullLogSpectrum);
  idft(fullLogSpectrum, invDFTLog, DFT_REAL_OUTPUT);
  showScaled("invDFTLog", invDFTLog);
  
  Mat filteredSpectrum;
  Mat avgFilter =
    Mat::ones(avgFilterSize, avgFilterSize, CV_32FC1) / (avgFilterSize * avgFilterSize);
  
  filter2D(logSpectrum, filteredSpectrum, -1, avgFilter);
  Mat residual = logSpectrum - filteredSpectrum;


  // turn the spectrum back to an image, with some filtering
  Mat exponentiated(residual.rows, residual.cols, CV_32FC2);
  Mat unfilteredOutput(downSampled.rows, downSampled.cols, CV_32FC2);
  Mat squared;

  // exponentiating the coefficients manually, because opencv sucks at complex 
  // arithmetic
  for (int i = 0; i < residual.rows; i++) {
    for (int j = 0; j < residual.cols; j++) {
      complex<float> coeff(residual.at<float>(i,j), parts[1].at<float>(i,j));
      complex<float> expCoeff = exp(coeff);
      
      exponentiated.at<Vec2f>(i,j)[0] = expCoeff.real();
      exponentiated.at<Vec2f>(i,j)[1] = expCoeff.imag();
    }
  }

  cout<<"running inverse dft"<<endl;
  idft(exponentiated, unfilteredOutput, DFT_SCALE);
  Mat unfilteredParts[2];
  split(unfilteredOutput, unfilteredParts);
  pow(unfilteredParts[0], 2, squared);
  GaussianBlur(squared, saliencyMap, Size(0,0), gaussianSigma);
}

