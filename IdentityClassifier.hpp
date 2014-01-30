#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

/**
 * Classifies animation character images according to character identity. Right
 * now, it's a dummy which always returns the same class label, intended for testing
 * purposes only.
 */
class IdentityClassifier {
public:
  IdentityClassifier();

  /**
   * Trains the classifier with a training set of image filenames with associated
   * class labels.
   *
   * @param trainingImageFilenames images to train the classifier with.
   */
  void train(
    vector<string> &trainingImageFilenames, 
    vector<int> &trainingClassLabels);

  /**
   * Predicts the identity of the character in an image. Will fail if the classifier
   * hasn't been trained prior to this method call (i.e. via the train or load
   * methods).
   *
   * @param testImageFilename filename to the character image to run prediction on.
   */
  int predict(string testImageFilename);

  /**
   * Saves the results of training the classifier to a file. Will fail if the 
   * classifier hasn't been trained prior to this method call (i.e. via the train 
   * or load methods).
   *
   * @param filename file to save training data to.
   */
  void save(string filename);

  /**
   * Load training data from a file. The file must have been created through a call
   * to the save method.
   *
   * @param filename file to load training data from.
   */
  void load(string filename);
};
