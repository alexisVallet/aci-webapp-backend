#pragma once
/**
 * Implementation of the spectral residual saliency detection algorithm by Hou et al.,
 * 2007.
 */
#include <opencv2/opencv.hpp>
#include <complex>
#include <vector>

using namespace cv;
using namespace std;

/**
 * Computes the saliency map of an input grayscale image using the spectral residual
 * method.
 *
 * @param image input image to detect saliencies from.
 * @param saliencyMap output saliency map of the input image.
 * @param avgFilterSize size of the average filter used to approximate the general shape
 * of the log spectra of the input image.
 * @param gaussianSigma sigma parameter to a final gaussian filter to apply to the
 * final saliency map.
 */
void spectralResidualSaliencyMap(
  const Mat_<float> &image, 
  Mat_<float> &saliencyMap, 
  int avgFilterSize = 3, 
  float gaussianSigma = 8,
  int downSamplingMaxDim = 64);

