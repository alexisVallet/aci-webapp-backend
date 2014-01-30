#pragma once
/**
 * Header file providing a C interface to the library functions, 
 * intended for use with the Haskell FFI .
 */

/*
 * Boilerplate so the header file can be interpreted as both a C++ and C header
 * file. Makes it easier to call from the Haskell FFI.
 */
#ifdef __cplusplus
extern "C" {
#endif
  typedef void *IdentityClassifier_ptr;

  /**
   * Creates and trains an identity classifier with specific images and class
   * labels.
   *
   * @param trainingSamples array of samples to train the classifier with, specified
   * by image filename and corresponding class label.
   * @param nbSamples number of training samples, must be equal to the length of
   * the trainingSamples array.
   * @return dynamically allocated pointer to the classifier. Must be freed using
   * freeIdentityClassifier.
   */
  extern IdentityClassifier_ptr trainIdentity(
    char **trainingImageFilenames,
    int *trainingClassLabels,
    int nbSamples);

  /**
   * Predicts the identity of the character in an image using the specified
   * classifier.
   *
   * @param classifier classifier to predict the identity of the character with.
   * @param testImageFilename file name of the character image to predict the
   * identity of.
   * @return the predicted class label of the image.
   */
  extern int predictIdentity(
    IdentityClassifier_ptr classifier,
    char *testImageFilename);

  /**
   * Saves the classifier to a file.
   *
   * @param classifier classifier to write to a file.
   * @param filename file name to write the classifier to.
   */
  extern void saveIdentityClassifier(
    IdentityClassifier_ptr classifier,
    char *filename);

  /**
   * Loads a trained classifier from a file. The file must have been
   * created by a call to saveIdentityClassifier.
   *
   * @param filename file to load the classifier from.
   * @return the classifier trained with the data in the file.
   */
  extern IdentityClassifier_ptr loadIdentityClassifier(
    char *filename);

  /**
   * Frees the resources used by the specificed classifier.
   *
   * @param classifier classifier to free the resources of. This pointer should
   * not be used after freeing, unspecified behavior will arise otherwise.
   */
  extern void freeIdentityClassifier(
    IdentityClassifier_ptr classifier);
#ifdef __cplusplus
}
#endif
