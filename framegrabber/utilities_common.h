#ifndef __DLBP_TRAINING_UTILITIES_COMMON_WFL__
#define __DLBP_TRAINING_UTILITIES_COMMON_WFL__

#include <string>
#include <vector>

using namespace std;

void tic();
double toc();

void GetFilePaths(const string folderPath, const vector<string>& exts, vector<string>& filePaths, bool append = false);
void GetFilePaths(const string folderPath, const char* _exts, vector<string>& filePaths, bool append = false);

string BaseName(const string& path);


#endif
