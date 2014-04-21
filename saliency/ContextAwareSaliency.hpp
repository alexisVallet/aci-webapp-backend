#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Returns a saliency map of a color image showing the object of interest
 * in the image. Implements the context aware saliency algorithm by
 * Zelnik-Manor, 2012.
 *
 * @param image input BGR image (as returned by cv::imread) to compute 
 * the saliency map for.
 * @param saliency output saliency map for the input image. Larger values
 * means more salient pixels. Resized so its maximum dimension is given
 * by the maxDimension parameter.
 * @param scales the set of patch sizes to consider for each pixel (set R in
 * the original paper), in pixels of the dimension of the square patch.
 * @param multiscales factor of the scale of patches to consider for what the
 * paper calls "multi scale saliency enhancement".
 * @param k number of nearest patch to consider for each pixel. Higher values
 * yield higher quality saliency maps, lower values yield faster runtime.
 * @param maxDimension maximum dimension of the output saliency map. Higher
 * values yield higher quality saliency maps, lower values yield faster runtime.
 */
void contextAwareSaliency(const Mat_<Vec3b> &image, Mat_<float> &saliency, vector<int> &scales, vector<float> &multiScales, int k = 64, int maxDimension = 250);
