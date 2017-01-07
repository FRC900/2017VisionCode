#pragma once

#ifdef USE_GIE
#include "Infer.h"
#endif
#include "Classifier.hpp"

template <class MatT>
class GIEClassifier : public Classifier<MatT>
{
	public:
		GIEClassifier(const std::string& modelFile,
					const std::string& trainedFile,
					const std::string& zcaWeightFile,
					const std::string& labelFile,
					const size_t batchSize);
		~GIEClassifier();

		bool initialized(void) const;

	private:
#ifdef USE_GIE
		// Wrap input layer of the net into separate Mat objects
		// This sets them up to be written with actual data
		// in PreprocessBatch()
		void WrapBatchInputLayer(void);
#endif

		// Get the output values for a set of images
		// These values will be in the same order as the labels for each
		// image, and each set of labels for an image next adjacent to the
		// one for the next image.
		// That is, [0] = value for label 0 for the first image up to 
		// [n] = value for label n for the first image. It then starts again
		// for the next image - [n+1] = label 0 for image #2.
		std::vector<float> PredictBatch(const std::vector<MatT> &imgs);

	private:
#ifdef USE_GIE
		// TODO : try shared pointers
		nvinfer1::IRuntime* runtime_;          // network runtime
		nvinfer1::ICudaEngine* engine_;        // network engine
		nvinfer1::IExecutionContext *context_; // netowrk context to run on engine
		cudaStream_t stream_;

		void* buffers_[2];           // input and output GPU buffers
		size_t inputIndex_;
		size_t outputIndex_;
		size_t numChannels_;

		float *inputCPU_;            // input CPU buffer
		std::vector<std::vector<MatT>> inputBatch_; // net input buffers wrapped in Mats
#endif

		bool initialized_;
};
