// Class for performing zero-phase component analysis & transforms. 
// The goal is to transform input such that the data is zero-mean,
// that individual features are uncorrelated and all have the 
// same variance. This process is known as whitening or sphering. 
// Whitening takes cues from so-called white noise, where the
// next value in a series can't be predicted from previous 
// sample. Sphering refers to the appearance of graphs of the 
// processed data, which look like spherical blobs centered around
// the origin.
// ZCA is a specific type of principal component analysis which 
// performs this transformation with the minimum amount of change
// to the original data. Thus ZCA-whitened images resemble 
// the original input in some sense - this isn't true for the
// general case of possible PCA whitening.
// PCA takes a set of data and transforms it so that the
// variables are uncorrelated.  This transformation takes an N
// dimensional set of data and rotates it to a new coordinate system
// The first axis of the rotated data will be chosen such that
// it accounts for the largest variation in the data. 
// Subsquent dimensions are ones which account for less and 
// less variation.  Each axis is called a principal component,
// and has both a unit vector (where that axis points) and a separate
// magnitude showing how much that particular component impacts the
// overall data value.
// Imagine a 2d set of data which is essentaily y = x +/- a small noise
// component. That is, the data is centered around y=x with small variations
// above and below.  PCA of that data set would rotate it so the
// new primary axis would be the line y=x. The second axis would
// be orthoganl to that and would quantify how much noise each
// data point had - how far it was displaced from the new primary
// y=x axis.  
// Once PCA is applied to a data set, the data points are decorrelated
// with each other.  To achieve the second goal of uniform variance
// the magnitude of each principal component is rescaled by the 
// appropriate amount.
// In this application, the each color channel of each pixel is
// a dimension. So a 12x12 3-channel color image would have 12x12x3 =
// 432 dimensions. There are a number of correlations in natural
// images which whitening can eliminate - colors of nearby pixels
// tend to be similar, the channels of each pixel can be correlated
// due to lighting, and so on.  Removing those correlations which appear
// in every image lets neural net classifier focus on correlations
// that are indicitive of objects we're actually interested in.
//
// Additional references :
// http://ufldl.stanford.edu/wiki/index.php/PCA and following pages on that wiki
// http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
// http://eric-yuan.me/ufldl-exercise-pca-image/
// http://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
// Additional references :

#include <string>
#ifdef USE_MKL
#include <mkl.h>
#endif

#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include "portable_binary_oarchive.hpp"
#include "portable_binary_iarchive.hpp"
#include "cvMatSerialize.hpp"
#include "cvSizeSerialize.hpp"

#include "cuda_utils.hpp"
#include "zca.hpp"

using namespace std;
using namespace cv;
#if CV_MAJOR_VERSION == 2
using namespace cv::gpu;
#define cuda gpu
#elif CV_MAJOR_VERSION == 3
using namespace cv::cuda;
#endif

// Using the input images provided, generate a ZCA transform
// matrix.  Images will be resized to the requested size 
// before processing (the math doesn't work if input images
// are variable sized).  epsilon is a small positive value added
// to the magnitude of principal components before taking the
// square root of them - since many of these values are quite
// small the square root of them becomes numerically unstable.
ZCA::ZCA(const vector<Mat> &images, 
		 const Size &size, 
		 float epsilon) :
	size_(size),
	dPssIn_(NULL),
	epsilon_(epsilon),
	overallMin_(numeric_limits<double>::max()),
	overallMax_(numeric_limits<double>::min())
{
	if (images.size() == 0)
		return;

	// Build the transposed mat since that's the more
	// natural way opencv works - it is easier to add
	// new images as a new row but the math requires
	// them to columns. Since the later steps require 
	// both the normal and transposed versions of the
	// data, generate the transpose and then "untranspose"
	// it to get the correct one
	Mat workingMatT;

	// For each input image, convert to a floating point mat
	//  and resize to constant size
	// Find the mean and stddev of each color channel. Subtract
	// out the mean and divide by stddev to get to 0-mean
	// 1-stddev for each channel of the image - this 
	// helps normalize contrast between
	// images in different lighting conditions 
	// flatten to a single channel, 1 row matrix
	Mat resizeImg;
	Mat tmpImg;
	Scalar mean;
	Scalar stddev;
	for (auto it = images.cbegin(); it != images.cend(); ++it)
	{
		it->convertTo(resizeImg, CV_32FC3);
		cv::resize(resizeImg, tmpImg, size_);
		cv::meanStdDev(tmpImg, mean, stddev);
		cv::subtract(tmpImg, mean, tmpImg);
		cv::divide(tmpImg, stddev, tmpImg);
		workingMatT.push_back(tmpImg.reshape(1,1).clone());
	}

	// Transpose so each image is its own column 
	// rather than its own row 
	Mat workingMat(workingMatT.t());

	// sigma is the covariance matrix of the 
	// input data
	// Literature disagrees on dividing by cols or cols-1
	// Since we're using a large number of input
	// images it really doesn't matter that much
	Mat sigma = (workingMat * workingMatT) / (float)workingMat.cols;

	workingMatT.release();

	// Put this in a local scope so svd gets freed
	// after it isn't needed.
	{
		SVD svd;
		//Mat svdW; // eigenValues - magnitude of each principal component
		//Mat svdU; // eigenVectors - where each pricipal component points
		Mat svdVT;
		svd.compute(sigma, svdW_, svdU_, svdVT, SVD::FULL_UV);
	}
	
	//cout << "svdU" << endl << svdU_ << endl;
	//cout << "svdW" << endl << svdW_ << endl;

	// Add small epsilon to prevent sqrt(small number)
	// numerical instability. Larger epsilons have a
	// bigger smoothing effect
	// Take square root of each element, convert
	// from vector into diagonal array
	// Do the math on a local copy of svdW - save the original
	// in case we later want to re-do the calc using a differen
	// epsilon or whatever
	Mat svdW(svdW_.clone());
	svdW += epsilon_;
	cv::sqrt(svdW, svdW);
	svdW = 1.0 / svdW;
	Mat svdS(Mat::diag(svdW));

	// Weights are U * S * U'
	Mat weights = svdU_ * svdS * svdU_.t();

	// Transform the input images. Grab
	// a range of pixel values and use this
	// to convert back from floating point to
	// something in the range of 0-255
	// Don't want to use the full range of the
	// pixels since outliers will squash the range
	// most pixels end up in to just a few numbers.
	// Instead use the mean +/- 2.25 std deviations
	// This should allow full range representation of
	// > 96% of the pixels
	Mat transformedImgs = weights * workingMat;
	cv::meanStdDev(transformedImgs, mean, stddev);
	cout << "transformedImgs mean/stddev " << mean(0) << " " << stddev(0) << endl;
	overallMax_ = mean(0) + 2.25*stddev(0);
	overallMin_ = mean(0) - 2.25*stddev(0);

	// Formula to convert is uchar_val = alpha * float_val + beta
	// This will convert the majority of floating
	// point values into a 0-255 range that fits
	// into a normal 8UC3 mat without saturating
	cout << "Alpha / beta " << alpha() << " "<< beta() << endl;

	weightsT_ = weights.t();
	if (getCudaEnabledDeviceCount() > 0)
		weightsTGPU_.upload(weightsT_);
}

// Transform a typical 8 bit image as read from file
// Return the same 8UC3 type
// Just a wrapper around the faster version
// which does a batch at a time. Why aren't
// you using that one instead?
Mat ZCA::Transform8UC3(const Mat &input)
{
	vector<Mat> inputs;
	inputs.push_back(input);
	vector<Mat> outputs = Transform8UC3(inputs);
	return outputs[0].clone();
}

// Transform a typical 8 bit images as read from files
// Return the same 8UC3 type
vector<Mat> ZCA::Transform8UC3(const vector<Mat> &input)
{
	// If GPU is present do the transform
	// on the GPU
	if (getCudaEnabledDeviceCount() > 0)
	{
		GpuMat tmp;
		vector<GpuMat> f32List;
		for (auto it = input.cbegin(); it != input.cend(); ++it)
		{
			// Upload to GpuMat, convert to float array
			GpuMat(*it).convertTo(tmp, CV_32FC3);
			f32List.push_back(tmp.clone());
		}

		// Create result array in GPU memory
		float *dResult;
		const size_t resultSize = gm_.rows      *   // # of images per batch
								  size_.area()  *   // pixels / image channel
								  sizeof(float) *   // bytes / pixel
							  	  3;                // channels
		cudaSafeCall(cudaMalloc(&dResult, resultSize), "Transform8UC3 cudaMalloc");
		// Do the transform
		Transform32FC3(f32List, dResult);

		// Create space for results on CPU and copy
		// data from device into it
		float hResult[resultSize / sizeof(float)];
		cudaSafeCall(cudaMemcpy(hResult, dResult, resultSize, cudaMemcpyDeviceToHost), 
				     "Transform8UC3 cudaMemcpy");

		// Free device memory
		cudaSafeCall(cudaFree(dResult), "Transform8UC3 cudaFree");

		// Output is in order expected by Caffe code - 
		// each channel is contiguous.  3-channel images
		// are expected to have color channels interleaved
		// Rearrange things here :
		float *hPtr = hResult;

		Mat ml[3];  // separate channels
		Mat mF32;   // 3-channel float mat
		Mat mU8;    // 3-channel uint8 mat
		vector <Mat> result;
		for (size_t i = 0; i < f32List.size(); i++)
		{
			// Create a Mat wrapping the data from
			// each channel.  Then update the pointer to
			// move one image channel ahead (width * height
			// pixels) to get to the start of the
			// next color channel
			for (size_t j = 0; j < 3; j++)
			{
				ml[j] = Mat(size_, CV_32FC1, hPtr);
				hPtr += size_.area();
			}
			// Merge into a 3-channel mat
			merge(ml, 3, mF32);

			// See comment below about the need for
			// alpha and beta adjustments
			mF32.convertTo(mU8, CV_8UC3, alpha(), beta());
			result.push_back(mU8.clone());
		}
		return result;
	}

	// Non GPU-code
	vector<Mat> f32List;
	Mat tmp;

	// Create an intermediate vector of f32 versions
	// of the input image. 
	for (auto it = input.cbegin(); it != input.cend(); ++it)
	{
		it->convertTo(tmp, CV_32FC3);
		f32List.push_back(tmp.clone());
	}
	// Do the transform 
	vector <Mat> f32Ret = Transform32FC3(f32List);

	// Convert back to uchar array with correct 0 - 255 range
	// This turns it into a "normal" image file which
	// can be processed and visualized using typical
	// tools
	// The float version will have values in a range which
	// can't be exactly represented by the uchar version
	// (e.g. negative numbers, numbers larger than 255, etc).
	// Scaling by alpha/beta will shift the range of
	// float values to 0-255 (techinally, it'll move ~96%
	// of the value into that range, the rest will be saturated
	// to 0 or 255).  When training using these ucar images,
	// be sure to undo the scaling so they are converted back
	// to the correct float values
	vector <Mat> ret;
	for (auto it = f32Ret.cbegin(); it != f32Ret.cend(); ++it)
	{
		it->convertTo(tmp, CV_8UC3, alpha(), beta());
		ret.push_back(tmp.clone());
	}

	return ret;
}

// Expects a 32FC3 mat as input
// Just a wrapper around the faster version
// which does a batch at a time. Why aren't
// you using that one instead?
Mat ZCA::Transform32FC3(const Mat &input)
{
	vector<Mat> inputs;
	inputs.push_back(input);
	vector<Mat> outputs = Transform32FC3(inputs);
	return outputs[0].clone();
}

// Transform a vector of input images in floating
// point format using the weights loaded
// when this object was initialized
vector<Mat> ZCA::Transform32FC3(const vector<Mat> &input)
{
	Mat output;
	Mat work;
	// Create a large mat holding all of the pixels
	// from all of the input images.
	// Each row is data from one image. Each image
	// is flattened to 1 channel of interlaved B,G,R
	// values.  
	// Global contrast normalization is applied to
	// each image - subtract the mean and divide
	// by the standard deviation separately for each
	// channel. That way each image is normalized to 0-mean 
	// and a standard deviation of 1 before running it
	// through ZCA weights.
	for (auto it = input.cbegin(); it != input.cend(); ++it)
	{
		if (it->size() != size_)
			cv::resize(*it, output, size_);
		else 
			// need clone so mat is contiguous - 
			// reshape won't work otherwise
			output = it->clone();

		Scalar mean;
		Scalar stddev;
		cv::meanStdDev(output, mean, stddev);

		for (int r = 0; r < output.rows; r++)
		{
			Vec3f *p = output.ptr<Vec3f>(r);
			for (int c = 0; c < output.cols; c++)
				for (int ch = 0; ch < 3; ch++)
					p[c][ch] = (p[c][ch] - mean[ch]) / stddev[ch];
		}

		// Reshape flattens the image to 1 channel, 1 row.
		// Push that row into the bottom of work
		work.push_back(output.reshape(1,1));
	}

	// Apply ZCA transform matrix
	// Math here is weights * images = output images
	// This works if each image is a column of data
	// The natural way to add data above using push_back
	//  creates a transpose of that instead (i.e. each image is its
	//  own row rather than its own column).  Take advantage
	//  of the identiy (AB)^T = B^T A^T.  A=weights, B=images
	// Since we want to pull images apart in the same transposed
	// order, this saves a few transposes and gives a
	// slight performance bump.
#ifdef USE_MKL // Intel MKL libs speed up matrix math on Intel CPUs
	const size_t m = work.rows;
	const size_t n = weightsT_.rows;
	const size_t k = weightsT_.cols;

	output = Mat(work.rows, work.cols, CV_32FC1);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
			m, n, k, 1.0, (const float *)work.data, k, (const float *)weightsT_.data, n, 0.0, (float *)output.data, n);
#else
	cv::gemm(work, weightsT_, 1.0, Mat(), 0.0, output);
#endif

	// Matrix comes out transposed - instead
	// of an image per column it is an image per row.
	// That's a natural fit for taking them apart
	// back into images, though, so it save some time
	// not having to transpose the output

	// Each row is a different input image,
	// put them each into their own Mat
	vector<Mat> ret;
	for (int i = 0; i < output.rows; i++)
	{
		// Turn each row back into a 2-d mat with 3 float color channels
		ret.push_back(output.row(i).reshape(input[i].channels(), size_.height));
	}

	return ret;
}

void cudaZCATransform(const vector<GpuMat> &input, 
		const GpuMat &weightsT, 
		PtrStepSz<float> *dPssIn,
		GpuMat &gm,
		GpuMat &gmOut,
		float *output);

#if CV_MAJOR_VERSION == 3
#include <opencv2/cudawarping.hpp>
#endif
// Transform a vector of input images in floating
// point format using the weights loaded
// when this object was initialized
void ZCA::Transform32FC3(const vector<GpuMat> &input, float *dest)
{
	vector<GpuMat> foo;
	for (auto it = input.cbegin(); it != input.cend(); ++it)
	{
		if (it->size() != size_)
		{
			foo.push_back(GpuMat());
			cuda::resize(*it, foo[foo.size() - 1], size_);
		}
		else 
		{
			foo.push_back(*it);
		}
	}
	cudaZCATransform(foo, weightsTGPU_, dPssIn_, gm_, gmOut_, dest);
}

// Load a previously calcuated set of weights from file
ZCA::ZCA(const string &fileName, size_t batchSize) :
	dPssIn_(NULL)
{
	string ext = boost::filesystem::extension(fileName);
	if (getCudaEnabledDeviceCount() > 0)
		setDevice(0);
	// .zca files are a binary, compressed version of the
	// weights.  Load these if they exist since it is
	// quicker than loading the test-based XML file below
	if ((ext == ".zca") || (ext == ".ZCA"))
	{
		using namespace boost::iostreams;
		
		ifstream serializeIn(fileName, ios::in | ios::binary);
		if (!serializeIn.good())
		{
			cerr << "Could not open ifstream(" << fileName << ") for reading" << endl;
			return;
		}

		filtering_streambuf<input> filtSBIn;
		filtSBIn.push(zlib_decompressor());
		filtSBIn.push(serializeIn);
		portable_binary_iarchive portableArchiveIn(filtSBIn);
		portableArchiveIn >> *this;
	}
	else if ((ext == ".xml") || (ext == ".XML"))
	{
		try 
		{
			FileStorage fs(fileName, FileStorage::READ);
			if (fs.isOpened())
			{
				fs["ZCASize"] >> size_;
				fs["SVDU"] >> svdU_;
				fs["SVDW"] >> svdW_;
				Mat weights;
				fs["ZCAWeights"] >> weights;

				weightsT_ = weights.t();
				if (!weightsT_.empty() && (getCudaEnabledDeviceCount() > 0))
					weightsTGPU_.upload(weightsT_);

				fs["ZCAEpsilon"] >> epsilon_;
				fs["OverallMin"] >> overallMin_;
				fs["OverallMax"] >> overallMax_;
			}
			fs.release();
		}
		catch (const std::exception &e)
		{
			return;
		}
	}

	if (!weightsTGPU_.empty())
	{
		cudaSafeCall(cudaMalloc(&dPssIn_, batchSize * sizeof(PtrStepSz<float>)), "cudaMalloc dPssIn");
		gm_ = GpuMat(batchSize, size_.area() * 3, CV_32FC1);
		gmOut_ = GpuMat(gm_.size(), gm_.type());
	}
}

ZCA::ZCA(const ZCA &zca) :
	size_(zca.size_),
	svdU_(zca.svdU_),
	svdW_(zca.svdW_),
	weightsT_(zca.weightsT_),
	dPssIn_(NULL),
	epsilon_(zca.epsilon_),
	overallMin_(zca.overallMin_),
	overallMax_(zca.overallMax_)
{
	if (!weightsT_.empty() && (getCudaEnabledDeviceCount() > 0))
	{
		size_t batchSize = zca.gm_.rows;
		weightsTGPU_.upload(weightsT_);
		cudaSafeCall(cudaMalloc(&dPssIn_, batchSize * sizeof(PtrStepSz<float>)), "cudaMalloc dPssIn");
		gm_ = zca.gm_.clone();
		gmOut_ = zca.gm_.clone();
	}
}

ZCA &ZCA::operator=(const ZCA &zca)
{
	if (this != &zca)
	{
		size_ = zca.size_;
		svdU_ = zca.svdU_;
		svdW_ = zca.svdW_;
		weightsT_ = zca.weightsT_;
		if (dPssIn_)
		{
			cudaSafeCall(cudaFree(dPssIn_), "cudaFreedPssIn");
			dPssIn_ = NULL;
		}
		epsilon_ = zca.epsilon_;
		overallMin_ = zca.overallMin_;
		overallMax_ = zca.overallMax_;
		if (!weightsT_.empty() && (getCudaEnabledDeviceCount() > 0))
		{
			size_t batchSize = zca.gm_.rows;
			weightsTGPU_.upload(weightsT_);
			cudaSafeCall(cudaMalloc(&dPssIn_, batchSize * sizeof(PtrStepSz<float>)), "cudaMalloc dPssIn");
			gm_ = zca.gm_.clone();
			gmOut_ = zca.gm_.clone();
		}
	}
	return *this;
}

ZCA::~ZCA()
{
	if (dPssIn_)
		cudaSafeCall(cudaFree(dPssIn_), "cudaFree dPssIn");
}

// Remove the ZCA weights / eigenvectors with
// the lowest weights
void ZCA::Resize(int size)
{	
	Mat weights(weightsT_.clone());

	Mat svdW(svdW_(Rect(0,0,1,size)).clone());
	cout << "svdW_" << svdW_.size() << endl;
	cout << "svdW" << svdW.size() << endl;
	svdW += epsilon_;
	cv::sqrt(svdW, svdW);
	svdW = 1.0 / svdW;
	Mat svdS(Mat::diag(svdW));
	cout << "svdS" << svdS.size() << endl;

	cout << "svdU_" << svdU_.size() << endl;
	Mat svdU(svdU_(Rect(0,0,size,svdU_.rows)).clone());
	cout << "svdU" << svdU.size() << endl;

	// Weights are U * S * U'
	weights = svdU * svdS * svdU.t();
	weightsT_ = weights.t();
	if (getCudaEnabledDeviceCount() > 0)
		weightsTGPU_.upload(weightsT_);
}

// Save calculated weights to a human-readble XML file
void ZCA::Write(const string &fileName) const
{
	FileStorage fs(fileName, FileStorage::WRITE);
	fs << "ZCASize" << size_;
	fs << "SVDU" << svdU_;
	fs << "SVDW" << svdW_;
	fs << "ZCAWeights" << weightsT_.t();
	fs << "ZCAEpsilon" << epsilon_;
	fs << "OverallMin" << overallMin_;
	fs << "OverallMax" << overallMax_;
}

//Serialization support for ZCA class
template <class Archive>
void ZCA::save(Archive &ar, const unsigned int version) const
{
	(void)version;
	ar & size_;
	ar & svdU_;
	ar & svdW_;
	Mat weights(weightsT_.t());
	ar & weights;
	ar & epsilon_;
	ar & overallMax_;
	ar & overallMin_;
}

template <class Archive>
void ZCA::load(Archive &ar, const unsigned int version)
{
	(void)version;
	ar & size_;
	ar & svdU_;
	ar & svdW_;
	Mat weights;
	ar & weights;
	weightsT_ = weights.t();
	if (!weightsT_.empty() && (getCudaEnabledDeviceCount() > 0))
		weightsTGPU_.upload(weightsT_);
	ar & epsilon_;
	ar & overallMax_;
	ar & overallMin_;
}

// Save calculated weights to a compressed binary file
void ZCA::WriteCompressed(const string &fileName) const
{
	using namespace boost::iostreams;

	ofstream serializeOut(fileName, ios::out | ios::binary);
	if (!serializeOut.is_open())
	{
		cerr << "Could not open ofstream(" << fileName << ") for writing" << endl;
		return;
	}

	// Create a pipeline which takes a stream and compresses
	// it before writing it out to a file
	filtering_streambuf<output> filtSBOut;

	filtSBOut.push(zlib_compressor(zlib::best_speed));
	filtSBOut.push(serializeOut);

	// Create an output archive which writes to the previously
	// created output chain (zlib->output file path)
	portable_binary_oarchive archiveOut(filtSBOut);
	archiveOut << *this;
}

void ZCA::Print(void) const
{
	Mat weights(weightsT_.t());

	for (int r = 0; r < weights.rows; r++)
	{
		const float *ptr = weights.ptr<float>(r);
		for (int c = 0; c < weights.cols; c++)
			cout << *ptr++ << " ";
		cout << endl;
	}
}

// Generate constants to convert from float
// mat back to 8UC3 one.  
double ZCA::alpha(int maxPixelValue) const
{
	double range = overallMax_ - overallMin_;

	return maxPixelValue / range;
}
double ZCA::beta(void) const
{
	return -overallMin_ * alpha();
}

Size ZCA::size(void) const
{
	return size_;
}
