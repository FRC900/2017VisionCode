#pragma once
#include <opencv2/opencv.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
 
BOOST_SERIALIZATION_SPLIT_FREE(::cv::Size)
namespace boost 
{
	namespace serialization 
	{
		/** Serialization support for cv::Size */
		template <class Archive>
		void save(Archive &ar, const ::cv::Size &s, const unsigned int version)
		{
			(void)version;
			ar & s.width;
			ar & s.height;
		}

		/** Serialization support for cv::Size */
		template <class Archive>
		void load(Archive &ar, ::cv::Size &s, const unsigned int version)
		{
			(void)version;
			int width;
			int height;

			ar & width;
			ar & height;

			s = ::cv::Size(width, height);
		}
	}
}
