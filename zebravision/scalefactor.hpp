#ifndef INC_SCALEFACTOR_HPP__
#define INC_SCALEFACTOR_HPP__

#include <vector>

template <class MatT>
void scalefactor(const MatT &inputimage, const cv::Size &objectsize,
      const cv::Size &minsize, const cv::Size &maxsize, double scaleFactor,
      std::vector<std::pair<MatT, double> > &ScaleInfo);

template <class MatT>
void scalefactor(const MatT &inputimage, 
		const std::vector<std::pair<MatT, double> > &scaleInfoIn,
		int rescaleFactor,
		std::vector<std::pair<MatT, double> > &scaleInfoOut);

#endif
