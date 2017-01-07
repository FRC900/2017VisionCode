#ifdef USE_GIE
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/core.hpp>
#include "caffeParser.h"
#include "GIEClassifier.hpp"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;
using namespace cv;
using namespace cv::gpu;

#define CHECK_CUDA(status)								\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

// stuff we know about the network and the caffe input/output blobs
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "softmax";

// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 std::stringstream &gieModelStream) // serialized version of model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	CaffeParser* parser = new CaffeParser;
	bool useFp16 = builder->plaformHasFastFp16();
	nvinfer1::DataType modelDataType = useFp16 ?nvinfer1:: DataType::kHALF : nvinfer1::DataType::kFLOAT; // create a 16-bit model if it's natively supported
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
															  modelFile.c_str(),
															  *network,
															  nvinfer1::DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine_
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	delete parser;

	// serialize the engine_, then close everything down
	engine->serialize(gieModelStream);
	engine->destroy();
	builder->destroy();
}


template <class MatT>
GIEClassifier<MatT>::GIEClassifier(const string& modelFile,
      const string& trainedFile,
      const string& zcaWeightFile,
      const string& labelFile,
      const size_t  batchSize) :
	Classifier<MatT>(modelFile, trainedFile, zcaWeightFile, labelFile, batchSize),
	numChannels_(3),
	inputCPU_(NULL),
	initialized_(false)
{
	if (!Classifier<MatT>::initialized())
		return;

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
	cout << "Loading GIE model " << endl << modelFile << endl << "\t" << trainedFile << endl << "\t" << zcaWeightFile << endl << "\t" << labelFile << endl;

	this->batchSize_ = batchSize;

	std::stringstream gieModelStream;
	// TODO :: read from file if exists and is newer than modelFile and trainedFile
	caffeToGIEModel(modelFile, trainedFile, std::vector <std::string>{OUTPUT_BLOB_NAME}, batchSize, gieModelStream);

	// Create runable version of model by
	// deserializing the engine 
	gieModelStream.seekg(0, gieModelStream.beg);

	runtime_ = createInferRuntime(gLogger);
	engine_  = runtime_->deserializeCudaEngine(gieModelStream);
	context_ = engine_->createExecutionContext();

	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine_->getNbBindings() == 2);

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	inputIndex_  = engine_->getBindingIndex(INPUT_BLOB_NAME); 
	outputIndex_ = engine_->getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK_CUDA(cudaMalloc(&buffers_[inputIndex_], batchSize * numChannels_ * this->inputGeometry_.area() * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&buffers_[outputIndex_], batchSize * this->labels_.size() * sizeof(float)));

	CHECK_CUDA(cudaStreamCreate(&stream_));

	// Set up input buffers for net
	WrapBatchInputLayer();

	initialized_ = true;
}

template <class MatT>
GIEClassifier<MatT>::~GIEClassifier()
{
	if (inputCPU_)
		delete [] inputCPU_;

	// release the stream and the buffers
	cudaStreamDestroy(stream_);
	CHECK_CUDA(cudaFree(buffers_[inputIndex_]));
	CHECK_CUDA(cudaFree(buffers_[outputIndex_]));
	context_->destroy();
	engine_->destroy();
	runtime_->destroy();
}

template <class MatT>
bool GIEClassifier<MatT>::initialized(void) const
{
	if (!Classifier<MatT>::initialized())
		return false;

	return initialized_;
}

// Wrap input layer of the net into separate Mat objects
// This sets them up to be written with actual data
// in PreprocessBatch() using OpenCV split() calls
template <class MatT>
void GIEClassifier<MatT>::WrapBatchInputLayer(void)
{
	if (inputCPU_)
		delete [] inputCPU_;
	inputCPU_= new float[this->batchSize_ * numChannels_ * this->inputGeometry_.area()];
	float *inputCPU = inputCPU_;

	inputBatch_.clear();

	for (size_t j = 0; j < this->batchSize_; j++)
	{
		vector<MatT> inputChannels;
		for (size_t i = 0; i < numChannels_; ++i)
		{
			MatT channel(this->inputGeometry_.height, this->inputGeometry_.width, CV_32FC1, inputCPU);
			inputChannels.push_back(channel);
			inputCPU += this->inputGeometry_.area(); // point to start of next image's location in the buffer
		}
		inputBatch_.push_back(vector<MatT>(inputChannels));
	}
}

template <>
vector<float> GIEClassifier<Mat>::PredictBatch(const vector<Mat> &imgs)
{
	if (imgs.size() > this->batchSize_) 
		cerr <<
		"PreprocessBatch() : too many input images : batch size is " << 
		this->batchSize_ << "imgs.size() = " << imgs.size() << endl; 

	// Take each image in Mat, convert it to the correct image type,
	// color depth, size to match the net input. Convert to 
	// F32 type, since that's what the net inputs are. 
	// Subtract out the mean before passing to the net input
	// Then actually write the images to the net input memory buffers
	vector<Mat> zcaImgs = this->zca_.Transform32FC3(imgs);
	for (size_t i = 0 ; i < zcaImgs.size(); i++)
	{
		/* This operation will write the separate BGR planes directly to the
		 * inputCPU_ array since it is wrapped by the MatT
		 * objects in inputChannels. */
		vector<Mat> *inputChannels = &inputBatch_.at(i);
		split(zcaImgs[i], *inputChannels);
	}
	float output[this->labels_.size() * this->batchSize_];
	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK_CUDA(cudaMemcpyAsync(buffers_[inputIndex_], inputCPU_, this->batchSize_ * this->inputGeometry_.area() * sizeof(float), cudaMemcpyHostToDevice, stream_));
	context_->enqueue(this->batchSize_, buffers_, stream_, nullptr);
	CHECK_CUDA(cudaMemcpyAsync(output, buffers_[outputIndex_], this->batchSize_ * this->labels_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
	cudaStreamSynchronize(stream_);

	return vector<float>(output, output + sizeof(output)/sizeof(output[0]));
}

template <>
vector<float> GIEClassifier<GpuMat>::PredictBatch(const vector<GpuMat> &imgs)
{
	if (imgs.size() > this->batchSize_) 
		cerr <<
		"PreprocessBatch() : too many input images : batch size is " << 
		this->batchSize_ << "imgs.size() = " << imgs.size() << endl; 

	// Take each image in Mat, convert it to the correct image type,
	// color depth, size to match the net input. Convert to 
	// F32 type, since that's what the net inputs are. 
	// Subtract out the mean before passing to the net input
	// Then actually write the images to the net input memory buffers
	this->zca_.Transform32FC3(imgs, (float *)buffers_[inputIndex_]);

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK_CUDA(cudaMemcpyAsync(buffers_[inputIndex_], inputCPU_, this->batchSize_ * this->inputGeometry_.area() * sizeof(float), cudaMemcpyHostToDevice, stream_));
	context_->enqueue(this->batchSize_, buffers_, stream_, nullptr);

	// Setup an output buffer and enqueue a copy
	// into it once the net has been run
	float output[this->labels_.size() * this->batchSize_];
	CHECK_CUDA(cudaMemcpyAsync(output, buffers_[outputIndex_], this->batchSize_ * this->labels_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
	cudaStreamSynchronize(stream_);

	return vector<float>(output, output + sizeof(output)/sizeof(output[0]));
}
#else
#include <vector>
#include "opencv2_3_shim.hpp"
#include "GIEClassifier.hpp"

using namespace std;
using namespace cv;

template <class MatT>
GIEClassifier<MatT>::GIEClassifier(const string& modelFile,
      const string& trainedFile,
      const string& zcaWeightFile,
      const string& labelFile,
      const size_t  batchSize) :
	Classifier<MatT>(modelFile, trainedFile, zcaWeightFile, labelFile, batchSize),
	initialized_(false)
{
	cerr << "GIE support not available" << endl;
}

template <class MatT>
GIEClassifier<MatT>::~GIEClassifier()
{
}

template <class MatT>
bool GIEClassifier<MatT>::initialized(void) const
{
	return true;
}

template <class MatT>
vector<float> GIEClassifier<MatT>::PredictBatch(const vector<MatT> &imgs)
{
	cerr << "GIE support not available" << endl;
	return vector<float>(imgs.size() * this->labels_.size(), 0.0);
}
#endif

#if 0
int main(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
	caffeToGIEModel("mnist.prototxt", "mnist.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);

	// read a random digit file
	srand(unsigned(time(nullptr)));
	uint8_t fileData[INPUT_H*INPUT_W];
	readPGMFile(std::to_string(rand() % 10) + ".pgm", fileData);

	// print an ascii representation
	std::cout << "\n\n\n---------------------------" << "\n\n\n" << std::endl;
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

	// parse the mean file and 	subtract it from the image
	IBinaryProtoBlob* meanBlob = CaffeParser::parseBinaryProto(locateFile("mnist_mean.binaryproto").c_str());
	const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

	float data[INPUT_H*INPUT_W];
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		data[i] = float(fileData[i])-meanData[i];

	meanBlob->destroy();

	// deserialize the engine 
	gieModelStream.seekg(0, gieModelStream.beg);

	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream);

	IExecutionContext *context = engine->createExecutionContext();

	// run inference
	float prob[OUTPUT_SIZE];
	doInference(*context, data, prob, 1);

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	// print a histogram of the output distribution
	std::cout << "\n\n";
	for (unsigned int i = 0; i < 10; i++)
		std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
	std::cout << std::endl;

	return 0;
}
#endif

template class GIEClassifier<Mat>;
template class GIEClassifier<GpuMat>;
