#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "opencv2_3_shim.hpp"

#include "Classifier.hpp"

using namespace std;
using namespace cv;

// Simple test to see if a file exists and
// is accessable
template <class MatT>
bool Classifier<MatT>::fileExists(const string &fileName) const
{
	struct stat statBuffer;
	return stat(fileName.c_str(), &statBuffer) == 0;
}

template <class MatT>
Classifier<MatT>::Classifier(const string& modelFile,
      const string& trainedFile,
      const string& zcaWeightFile,
      const string& labelFile,
      const size_t  batchSize) :
	batchSize_(batchSize),
	zca_(zcaWeightFile, batchSize)
{
	(void)modelFile;
	(void)trainedFile;
	if (!fileExists(zcaWeightFile))
	{
		cerr << "Could not find ZCA weight file " << zcaWeightFile << endl;
		return;
	}
	if (!fileExists(labelFile))
	{
		cerr << "Could not find label file " << labelFile << endl;
		return;
	}

	// Load labels
	// This will be used to give each index of the output
	// a human-readable name
	ifstream labels(labelFile.c_str());
	if (!labels) 
	{
		cerr << "Unable to open labels file " << labelFile << endl;
		return;
	}
	string line;
	while (getline(labels, line))
		labels_.push_back(string(line));

	inputGeometry_ = zca_.size();
}

template <class MatT>
Classifier<MatT>::~Classifier()
{
}

// Helper function for compare - used to sort values by pair.first keys
// TODO : redo as lambda
static bool PairCompare(const pair<float, int>& lhs, 
						const pair<float, int>& rhs) 
{
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static vector<int> Argmax(const vector<float>& v, size_t N) 
{
	vector<pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(make_pair(v[i], i));
	partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	vector<int> result;
	for (size_t i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

// Given X input images, return X vectors of predictions.
// Each of the X vectors are themselves a vector which will have the 
// N predictions with the highest confidences for the corresponding
// input image
template <class MatT>
vector< vector<Prediction> > Classifier<MatT>::ClassifyBatch(
		const vector<MatT> &imgs, const size_t numClasses)
{
	// outputBatch will be a flat vector of N floating point values 
	// per image (1 per N output labels), repeated
	// times the number of input images batched per run
	// Convert that into the output vector of vectors
	vector<float> outputBatch = PredictBatch(imgs);
	return floatsToPredictions(outputBatch, imgs.size(), numClasses);
}

template <class MatT>
vector<vector<Prediction>> Classifier<MatT>::floatsToPredictions(const vector<float> &floats, const size_t imgSize, const size_t numClasses)
{
	vector< vector<Prediction> > predictions;
	const size_t labelsSize = labels_.size();
	const size_t classes = min(numClasses, labelsSize);
	// For each image, find the top numClasses values
	for(size_t j = 0; j < imgSize; j++)
	{
		// Create an output vector just for values for this image. Since
		// each image has labelsSize values, that's floats[j*labelsSize]
		// through floats[(j+1) * labelsSize]
		vector<float> output(floats.begin() + j*labelsSize, floats.begin() + (j+1)*labelsSize);
		// For the output specific to the jth image, grab the
		// indexes of the top classes predictions
		vector<int> maxN = Argmax(output, classes);
		// Using those top N indexes, create a set of labels/value predictions
		// specific to this jth image
		vector<Prediction> predictionSingle;
		for (size_t i = 0; i < classes; ++i) 
		{
			int idx = maxN[i];
			predictionSingle.push_back(make_pair(labels_[idx], output[idx]));
		}
		// Add the predictions for this image to the list of
		// predictions for all images
		predictions.push_back(vector<Prediction>(predictionSingle));
	}
	return predictions;
}

// Assorted helper functions
template <class MatT>
size_t Classifier<MatT>::batchSize(void) const
{
	return batchSize_;
}

template <class MatT>
Size Classifier<MatT>::getInputGeometry(void) const
{
	return inputGeometry_;
}

template <class MatT>
bool Classifier<MatT>::initialized(void) const
{
	if (labels_.size() == 0)
		return false;
	if (inputGeometry_ == Size())
		return false;
	return true;
}

template class Classifier<Mat>;
template class Classifier<GpuMat>;
