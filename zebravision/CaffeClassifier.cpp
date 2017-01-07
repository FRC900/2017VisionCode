#include <iostream>
#include <sys/stat.h>

#include "opencv2_3_shim.hpp"

#include "CaffeClassifier.hpp"

using namespace std;
using namespace caffe;
using namespace cv;

// Google logging init stuff needs to happen
// just once per program run.  Use this
// var to make sure it does.
static bool glogInit_ = false;

#if 0
#include <sys/time.h>
static double gtod_wrapper(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#endif

template <class MatT>
CaffeClassifier<MatT>::CaffeClassifier(const string& modelFile,
      const string& trainedFile,
      const string& zcaWeightFile,
      const string& labelFile,
      const size_t  batchSize) :
	Classifier<MatT>(modelFile, trainedFile, zcaWeightFile, labelFile, batchSize),
	initialized_(false)
{
	// Base class loads labels and ZCA preprocessing data.
	// If those fail, bail out immediately.
	if (!Classifier<MatT>::initialized())
		return;

	// Make sure the model definition and 
	// weight files exist
	if (!this->fileExists(modelFile))
	{
		cerr << "Could not find Caffe model " << modelFile << endl;
		return;
	}
	if (!this->fileExists(trainedFile))
	{
		cerr << "Could not find Caffe trained weights " << trainedFile << endl;
		return;
	}

	cout << "Loading Caffe model " << modelFile << endl << "\t" << trainedFile << endl << "\t" << zcaWeightFile << endl << "\t" << labelFile << endl;

	// Switch to CPU or GPU mode depending on
	// which version of the class we're running
	Caffe::set_mode(IsGPU() ? Caffe::GPU : Caffe::CPU);

	// Hopefully this turns off any logging
	// Run only once the first time and CaffeClassifer
	// class is created
	if (!glogInit_)
	{
		::google::InitGoogleLogging("");
		::google::LogToStderr();
		::google::SetStderrLogging(3);
		glogInit_ = true;
	}

	// Load the network - this includes model 
	// geometry and trained weights
	net_.reset(new Net<float>(modelFile, TEST));
	net_->CopyTrainedLayersFrom(trainedFile);

	// Some basic checking to make sure life makes sense
	// Protip - it never really does
	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	// More sanity checking. Number of input channels
	// should be 3 since we're using color images.
	Blob<float>* inputLayer = net_->input_blobs()[0];
	const int numChannels = inputLayer->channels();
	CHECK(numChannels == 3) << "Input layer should have 3 channels.";

	// Also, make sure the input geometry matches
	// the size expected by the preprocessing filters
	if (this->inputGeometry_ != Size(inputLayer->width(), inputLayer->height()))
	{
		cerr << "Net size != ZCA size" << endl;
		return;
	}

	// Quick check to make sure there are enough labels
	// for each output
	Blob<float>* outputLayer = net_->output_blobs()[0];
	CHECK_EQ(this->labels_.size(), outputLayer->channels())
		<< "Number of labels is different from the output layer dimension.";

	// Set the network up for the specified batch
	// size. This processes a number of images in 
	// parallel. This works out to be a bit quicker
	// than doing them one by one
	inputLayer->Reshape(batchSize, numChannels,
						inputLayer->height(),
						inputLayer->width());

	// Forward dimension change to all layers
	net_->Reshape();

	// The wrap code puts the buffer for each individual channel
	// input to the net (one color channel of one image) into 
	// a separate Mat 
	// The inner vector here will be one Mat per channel of the 
	// input to the net. The outer vector is a vector of those
	// one for each of the batched input images.
	// This allows an easy copy from the input images
	// into the input buffers for the net by simply doing
	// an OpenCV split() call to split a 3-channel input
	// image into 3 separate 1-channel images arranged in
	// the correct order
	// 
	// GPU code path writes directly to the mutable_gpu_data()
	// pointer so no wrapping is needed
	if (!IsGPU())
	{
		Blob<float>* inputLayer = net_->input_blobs()[0];

		const size_t width  = inputLayer->width();
		const size_t height = inputLayer->height();
		const size_t num    = inputLayer->num();

		float* inputData = inputLayer->mutable_cpu_data();

		inputBatch_.clear();

		for (size_t j = 0; j < num; j++)
		{
			vector<MatT> inputChannels;
			for (int i = 0; i < inputLayer->channels(); ++i)
			{
				MatT channel(height, width, CV_32FC1, inputData);
				inputChannels.push_back(channel);
				inputData += width * height;
			}
			inputBatch_.push_back(vector<MatT>(inputChannels));
		}
	}
	
	// We made it!
	initialized_ = true;
}


// TODO : this probably isn't needed
template <class MatT>
CaffeClassifier<MatT>::~CaffeClassifier()
{
}

// Get the output values for a set of images in one flat vector
// These values will be in the same order as the labels for each
// image, and each set of labels for an image next adjacent to the
// one for the next image.
// That is, [0] = value for label 0 for the first image up to 
// [n] = value for label n for the first image. It then starts again
// for the next image - [n+1] = label 0 for image #2.
template <class MatT>
vector<float> CaffeClassifier<MatT>::PredictBatch(const vector<MatT> &imgs) 
{
	// Process each image so they match the format
	// expected by the net, then copy the images
	// into the net's input buffers
	//double start = gtod_wrapper();
	PreprocessBatch(imgs);
	//cout << "PreprocessBatch " << gtod_wrapper() - start << endl;
	//start = gtod_wrapper();
	// Run a forward pass with the data filled in from above
	net_->Forward();
	//cout << "Forward " << gtod_wrapper() - start << endl;

	//start = gtod_wrapper();
	// Copy the output layer to a flat vector 
	// Use CPU data output unconditionally - it has
	// to end up back at the CPU eventually so do it
	// now ... just as good as any other time
	Blob<float>* outputLayer = net_->output_blobs()[0];
	const float* begin = outputLayer->cpu_data();
	const float* end = begin + outputLayer->channels()*imgs.size();
	//cout << "Output " << gtod_wrapper() - start << endl;
	return vector<float>(begin, end);
}

// Take each image in Mat, convert it to the correct image size,
// and apply ZCA whitening to preprocess the files
// Then actually write the images to the net input memory buffers
template <>
void CaffeClassifier<Mat>::PreprocessBatch(const vector<Mat> &imgs)
{
	CHECK(imgs.size() <= this->batchSize_) <<
		"PreprocessBatch() : too many input images : batch size is " << this->batchSize_ << "imgs.size() = " << imgs.size(); 

	vector<Mat> zcaImgs = this->zca_.Transform32FC3(imgs);

	// Hack to reset input layer to think that
	// data is on the CPU side.  Only really needed
	// when CPU & GPU operations are combined
	net_->input_blobs()[0]->mutable_cpu_data()[0] = 0;

	// For each image in the list, copy it to 
	// the net's input buffer
	for (size_t i = 0 ; i < zcaImgs.size(); i++)
	{
		/* This operation will write the separate BGR planes directly to the
		 * input layer of the network because it is wrapped by the MatT
		 * objects in inputChannels. */
		vector<Mat> *inputChannels = &inputBatch_.at(i);
		split(zcaImgs[i], *inputChannels);
	}
}

// Take each image in GpuMat, convert it to the correct image type,
// and apply ZCA whitening to preprocess the files
// The GPU input to the net is passed in to Transform32FC3 and
// that function copies its final results directly into the
// input buffers of the net.
template <>
void CaffeClassifier<GpuMat>::PreprocessBatch(const vector<GpuMat> &imgs)
{
	CHECK(imgs.size() <= this->batchSize_) <<
		"PreprocessBatch() : too many input images : batch size is " << this->batchSize_ << "imgs.size() = " << imgs.size(); 

	float* inputData = net_->input_blobs()[0]->mutable_gpu_data();
	this->zca_.Transform32FC3(imgs, inputData);
}


// Specialize these functions - the Mat one works
// on the CPU while the GpuMat one works on the GPU
template <>
bool CaffeClassifier<Mat>::IsGPU(void) const
{
	// TODO : change to unconditional false
	// eventually once things are debugged
	//return (getCudaEnabledDeviceCount() > 0);
	return false;
}

template <>
bool CaffeClassifier<GpuMat>::IsGPU(void) const
{
	return true;
}

template <class MatT>
bool CaffeClassifier<MatT>::initialized(void) const
{
	if (!Classifier<MatT>::initialized())
		return false;
	
	return initialized_;
}


// Instantiate both Mat and GpuMat versions of the Classifier
template class CaffeClassifier<Mat>;
template class CaffeClassifier<GpuMat>;
