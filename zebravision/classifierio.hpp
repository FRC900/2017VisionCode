#ifndef CLASSIFIERIO_HPP__
#define CLASSIFIERIO_HPP__

#include <string>
#include <vector>

class ClassifierIO
{
	public:
		ClassifierIO(const std::string &baseDir, int dirNum, int stageNum);
		bool findNextClassifierStage(bool increment);
		bool findNextClassifierDir(bool increment);
		std::vector<std::string> getClassifierFiles(void) const;
		std::string print(void) const;
	private:
		std::string getClassifierDir(void) const;
		bool createFullPath(const std::string &fileName, std::string &output) const;
		void setSnapshots(void);

		std::string baseDir_;
		int dirNum_;
		std::vector<int> snapshots_;
		size_t stageIdx_;
};

#endif
