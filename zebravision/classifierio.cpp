#include <iostream>
#include <sstream>
#include <string>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <boost/filesystem.hpp>

#include "classifierio.hpp"

using namespace std;
using namespace boost::filesystem;

// Default constructor : takes a baseDir, dirNum and stageNum
// combines baseDir and dirNum to create a path to load config info
// from. stageNum is the training stage in that file.
ClassifierIO::ClassifierIO(const string &baseDir, int dirNum, int stageNum) :
    baseDir_ (baseDir),
    dirNum_  (dirNum),
    stageIdx_(0)
{
	// First try to find a directory starting with baseName
	// which has a labels.txt file
	// This starts with baseName, then baseName_0, up through
	// baseName_99
	string outputString;
	if (createFullPath("labels.txt", outputString) || findNextClassifierDir(true))
	{
		setSnapshots();
		if (snapshots_.size() == 0)
		{
			string dirName(getClassifierDir());
			cerr << "ERROR: No snapshot files in classifier directory " << dirName << endl;
		}
		if (stageNum == -1)
		{
			// If stageNum == -1, load the highest numbered stage. This makes
			// sense as a default since later stages have trained for longer
			// and should be better?
			stageIdx_ = snapshots_.size() - 1;
		}
		else
		{
			// If not using the default
			// Find a close match to the requested snapshot number
			for (stageIdx_ = 0; stageIdx_ < (snapshots_.size()-1); stageIdx_ += 1)
			{
				if (snapshots_[stageIdx_] >= stageNum)
				{
					break;
				}
			}
		}
	}
	else
	{
		cerr << "ERROR: Failed to find valid classifier directory" << endl;
	}
}

// In the current dir, grab a list of the iteration numbers for
// each snapshot file.  
void ClassifierIO::setSnapshots(void)
{
	snapshots_.clear();

	string dir = getClassifierDir();
	path p (dir);

	directory_iterator end_itr;

	// cycle through the directory
	for (directory_iterator itr(p); itr != end_itr; ++itr)
	{
		// If it's not a directory, check it
		if (is_regular_file(itr->path())) 
		{
			// Push caffemodel numbers onto snapshots_ list
			string currentFile(itr->path().string());
			if ((currentFile.compare(dir.size()+1, 14, "snapshot_iter_") == 0)	&& 
			    (currentFile.compare(currentFile.size()-11, 11, ".caffemodel") == 0))
			{
				int num;
				istringstream(currentFile.substr(dir.size()+15, currentFile.size()-11)) >> num;
				snapshots_.push_back(num);
			}
		}
	}
	// sort ascending
	sort(snapshots_.begin(), snapshots_.end());
}


// using the current directory number, generate a filename for that dir
// if it exists - if it doesnt, return an empty string
string ClassifierIO::getClassifierDir() const
{
	string fullDir = baseDir_;
	// Special-case -1 to mean no suffix after directory name.
	if (dirNum_ != -1)
	{
		fullDir += "_" + to_string(dirNum_);
	}
    path p(fullDir);
    if (exists(p) && is_directory(p))
    {
        return p.string();
    }
	cerr << "ERROR: Invalid classifier directory: " << fullDir << endl;
	return string();
}


bool ClassifierIO::createFullPath(const string &fileName, string &output) const
{
	path tmpPath(getClassifierDir());
	tmpPath /= fileName;
	if (!exists(tmpPath) || !is_regular_file(tmpPath))
	{
		cerr << "ERROR: Failed to open " << tmpPath.string() << endl;
		return false;
	}
	output = tmpPath.string();
	return true;
}

vector<string> ClassifierIO::getClassifierFiles() const
{
    // Get 4 needed files in the following order:
    // 1. deploy.prototxt
    // 2. snapshot_iter_#####.caffemodel
    // 3. mean.binaryproto
    // 4. labels.txt
    vector<string> output;
	string outputString;

	if (createFullPath("deploy.prototxt", outputString))
	{
		output.push_back(outputString);

		if (createFullPath("snapshot_iter_" + to_string(snapshots_[stageIdx_]) + ".caffemodel", outputString))
		{
			output.push_back(outputString);

			if (createFullPath("zcaWeights.zca", outputString))
			{
				output.push_back(outputString);

				if (createFullPath("labels.txt", outputString))
				{
					output.push_back(outputString);
				}
			}
		}
	}

    return output;
}

// Find the next valid classifier. 
bool ClassifierIO::findNextClassifierStage(bool increment)
{
	int adder = increment ? 1 : -1;
	int num = stageIdx_ + adder;

	if ((num < 0) || (num >= (int)snapshots_.size()))
		return false;

	stageIdx_ = num;
	return true;
}

// Find the next valid classifier dir. If one is found, switch
// to the closest numbered iteration in the new directory
bool ClassifierIO::findNextClassifierDir(bool increment)
{
   int adder = increment ? 1 : -1; // count either up or down

   // Save old dir and stage in case no other
   // good ones are found
   int oldDirNum = dirNum_;
   int oldStageNum;
  
   if (snapshots_.size())
	   oldStageNum = snapshots_[stageIdx_];
   else
	   oldStageNum = numeric_limits<int>::max();

   // Iterate through possible stages
   for (dirNum_ += adder; (dirNum_ >= -1) && (dirNum_ <= 100); dirNum_ += adder)
   {
	   // If a directory is found ...
	   if (getClassifierDir() != string())
       {
		   // see if there are snapshots in there
		   setSnapshots();
		   if (snapshots_.size() > 0)
		   {
			   // If so, pick a stagenum close to the one in the previous dir
			   for (stageIdx_ = 0; stageIdx_ < (snapshots_.size()-1); stageIdx_ += 1)
				   if (snapshots_[stageIdx_] >= oldStageNum)
					   break;
			   return true;
		   }
       }
   }

   // If no other valid dir is found, reset back to the 
   // original dir
   dirNum_ = oldDirNum;
   setSnapshots();
   return false;
}

string ClassifierIO::print() const
{
   return path(getClassifierDir()).filename().string() +  ":" + to_string(snapshots_[stageIdx_]);
}
