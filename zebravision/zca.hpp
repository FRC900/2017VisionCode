#pragma once
#include <string>
#include <vector>
#include <boost/serialization/split_member.hpp>
#include "opencv2_3_shim.hpp"
#if CV_MAJOR_VERSION == 2
using cv::gpu::PtrStepSz;
#elif CV_MAJOR_VERSION == 3
using cv::cuda::PtrStepSz;
#endif

class ZCA
{
	public:
		// Given a set of input images, build ZCA weights
		// for a size() x 3channel input image
		ZCA(const std::vector<cv::Mat> &images, const cv::Size &size, float epsilon);

		// Init a zca transformer by reading from a file
		ZCA(const std::string &xmlFilename, size_t batchSize);

		// Copy constructor - needed since some pointers
		// are allocated in constructor
		ZCA(const ZCA &zca);
		ZCA &operator=(const ZCA &zca);

		~ZCA();

		// Save ZCA state to file
		void Write(const std::string &fileName) const;
		void WriteCompressed(const std::string &fileName) const;

		void Resize(int size); 

		// Apply ZCA transofrm to a single image in
		// 8UC3 format (normal imread) and 32FC3
		// format (3 channels of float input)
		cv::Mat Transform8UC3 (const cv::Mat &input);
		cv::Mat Transform32FC3(const cv::Mat &input);

		// Batch versions of above - much faster
		// especially if GPU can be used
		std::vector<cv::Mat> Transform8UC3 (const std::vector<cv::Mat> &input);
		std::vector<cv::Mat> Transform32FC3(const std::vector<cv::Mat> &input);
		std::vector<GpuMat> Transform32FC3(const std::vector<GpuMat> &input);
		void Transform32FC3(const std::vector<GpuMat> &input, float *dest);

		void Print(void) const;

		// a and b parameters for transforming
		// float pixel values back to 0-255
		// uchar data
		double alpha(int maxPixelVal = 255) const;
		double beta(void) const;

		cv::Size size(void) const;

	private:
		// Code to load and save data as binary archive
		// using boot::serialization
		friend class boost::serialization::access;
		BOOST_SERIALIZATION_SPLIT_MEMBER()
		// When the class Archive corresponds to an output archive, the
		// & operator is defined similar to <<.  Likewise, when the class Archive
		// is a type of input archive the & operator is defined similar to >>.
		template<class Archive> void save(Archive &ar, const unsigned int version) const;
		template<class Archive> void load(Archive &ar, const unsigned int version);

		cv::Size size_;

		// Eigenvalues and eignevectors of the ZCA-transformed
		// data. Might not need this - remove to save space?
		cv::Mat  svdU_;
		cv::Mat  svdW_;
		// The weights, stored in both
		// the CPU and, if available, GPU
		// Note that internally the transposed version
		// of the weights are used to optimize some 
		// calculations - see notes in zca.cpp
		cv::Mat  weightsT_;
		GpuMat   weightsTGPU_;
		// GPU buffers - more efficient to allocate
		// them once gloabally and reuse them
		GpuMat   gm_;
		GpuMat   gmOut_;
		GpuMat   buf_;

		PtrStepSz<float> *dPssIn_;

		float            epsilon_;
		double           overallMin_;
		double           overallMax_;
};
