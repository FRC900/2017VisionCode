#include <iostream>
#include <stack>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "utilities_common.h"

std::stack<int64> tictoc_stack;
namespace fs = ::boost::filesystem;

int g_nThreads = 1;

void tic(void) 
{
    tictoc_stack.push(cv::getTickCount());
}

double toc(void) 
{
	double duration = ( cv::getTickCount() - tictoc_stack.top()) / cv::getTickFrequency();
    std::cout << "Time elapsed: "
              << duration
              << "s" << std::endl;
    tictoc_stack.pop();
	return duration;
}

//void GetTimeString(char *str)
//{
//	SYSTEMTIME sys;
//	GetLocalTime(&sys);
//	sprintf(str, "%04d%02d%02dT%02d%02d%02d.%03u000+0800", (int)sys.wYear, (int)sys.wMonth, 
//		(int)sys.wDay, (int)sys.wHour, (int)sys.wMinute, (int)sys.wSecond, sys.wMilliseconds);
//}

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void GetFilePaths(const string folderPath, const vector<string>& exts, vector<string>& filePaths, bool append)
{ 
	const fs::path root(folderPath);
	if (!fs::exists(root)) return;
	if (!append)
	   filePaths.clear();
	if (fs::is_directory(root))
	{
		fs::recursive_directory_iterator it(root);
		fs::recursive_directory_iterator endit;
		while(it != endit) {
			if (!fs::is_regular_file(*it)) {
				++it;
				continue;
			}
			string ext = it->path().extension().string();
			bool isEffective = false;
			for (size_t i = 0; i < exts.size(); i++) {
				if (ext == exts[i]) {
					isEffective = true;
					break;
				}
			}
			if (isEffective)
				filePaths.push_back(it->path().string());			
			++it;
		}
	}
}

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
// The input _exts can be like ".jpg,.png,...."
void GetFilePaths(const string folderPath, const char* _exts, vector<string>& filePaths, bool append)
{ 
	char* pch;
	char str[4096];
	strcpy(str, _exts);
	pch = strtok(str, "|");
	vector<string> exts;
	while (pch != 0) {
		exts.push_back(string(pch));
		pch = strtok(NULL, "|");
	}
	GetFilePaths(folderPath, exts, filePaths, append);
}

string BaseName(const string& path)
{
#ifdef __linux__
	char *fname;
	fname = basename((char *)path.c_str());
#else
	char drive[4096], dir[4096], fname[4096], ext[4096];
	_splitpath(path.c_str(), drive, dir, fname, ext);
#endif
	return string(fname);
}
