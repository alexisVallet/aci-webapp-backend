#include "aci.h"
#include "IdentityClassifier.hpp"

IdentityClassifier_ptr trainIdentity(
  char **trainingImageFilenames,
  int *trainingClassLabels,
  int nbSamples) {
  IdentityClassifier *classifier = new IdentityClassifier();
  vector<string> samplesVector;
  vector<int> labelsVector(trainingClassLabels, trainingClassLabels + nbSamples);

  samplesVector.reserve(nbSamples);

  for (int i = 0; i < nbSamples; i++) {
    samplesVector.push_back(string(trainingImageFilenames[i]));
  }

  classifier->train(samplesVector, labelsVector);

  return (void*)classifier;
}

int predictIdentity(
  IdentityClassifier_ptr classifier,
  char *testImageFilename) {
  string filenameString(testImageFilename);

  return ((IdentityClassifier*)classifier)->predict(filenameString);
}

void saveIdentityClassifier(
  IdentityClassifier_ptr classifier,
  char *filename) {
  string filenameString(filename);

  ((IdentityClassifier*)classifier)->save(filenameString);
}

IdentityClassifier_ptr loadIdentityClassifier(
  char *filename) {
  IdentityClassifier *classifier = new IdentityClassifier();
  string filenameString(filename);
  classifier->load(filenameString);
  
  return (void*)classifier;
}

void freeIdentityClassifier(
  IdentityClassifier_ptr classifier) {
  delete (IdentityClassifier*)classifier;
}
