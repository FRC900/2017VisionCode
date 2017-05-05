#pragma once

#include <string>

class CascadeClassifierIO
{
   public:
		CascadeClassifierIO(std::string baseDir, int dirNum, int stageNum);
		std::string getClassifierDir(void) const;
		std::string getClassifierName(void) const;
		bool findNextClassifierStage(bool increment);
		bool findNextClassifierDir(bool increment);
		std::string print(void) const;
   private :
		std::string baseDir_;
		int dirNum_;
		int stageNum_;
};

