#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <string>

#include "chroma_key.hpp"
#include "image_warp.hpp"
#include "imageShift.hpp"
#include "opencv2_3_shim.hpp"
#if CV_MAJOR_VERSION == 2
#define cuda gpu
#else
#include <opencv2/cudawarping.hpp>
#endif

using namespace std;
using namespace cv;
//#define DEBUG

#if 1
//Values for purple screen:
static int g_h_max = 170;
static int g_h_min = 130;
static int g_s_max = 255;
static int g_s_min = 130;
static int g_v_max = 255;
static int g_v_min = 48;
#else
#if 1
//Values for blue screen:
static int g_h_max = 120;
static int g_h_min = 110;
static int g_s_max = 255;
static int g_s_min = 220;
static int g_v_max = 150;
static int g_v_min = 50;
#else
//Values for blue2 videos
static int g_h_max = 135;
static int g_h_min = 100;
static int g_s_max = 254;
static int g_s_min = 100;
static int g_v_max = 240;
static int g_v_min = 20;
#endif
#endif
static int    g_files_per  = 10;
static int    g_num_frames = 75;
static int    g_min_resize = 0;
static int    g_max_resize = 0; //no resizing for now
static float  g_noise      = 3.0;
static string g_outputdir  = ".";
static string g_bgfile     = "";
static Point3f g_maxrot(0,0,0);
static bool   g_do_shifts = true;

#ifdef __CYGWIN__
inline int
stoi(const wstring& __str, size_t *__idx = 0, int __base = 10)
{
    return __gnu_cxx::__stoa<long, int>(&std::wcstol, "stoi", __str.c_str(),
                                        __idx, __base);
}
#endif

template<typename T>
string IntToHex(T i)
{
    stringstream stream;

    stream << setfill('0') << setw(2) << hex << i;
    return stream.str();
}


string Behead(string my_string)
{
    size_t found = my_string.rfind("/");

    return my_string.substr(found + 1);
}


void usage(char *argv[])
{
    cout << "usage: " << argv[0] << " [-r RGBM RGBT] [-f frames] [-i files] [--min min] [--max max] filename1 [filename2...]" << endl << endl;
    cout << "-r          RGBM and RGBT are hex colors RRGGBB; M is the median value and T is the threshold above or below the median" << endl;
    cout << "-f          frames is the number of frames grabbed from a video" << endl;
    cout << "-i          files is the number of output image files per frame" << endl;
    cout << "-o          change output directory from cwd/images to [option]/images" << endl;
    cout << "--min       min is the minimum percentage (as a decimal) for resizing for detection" << endl;
    cout << "--max       max is the max percentage (as a decimal) for resizing for detection" << endl;
	cout << "--maxxrot   max random rotation in x axis (radians)" << endl;
	cout << "--maxyrot   max random rotation in y axis (radians)" << endl;
	cout << "--maxzrot   max random rotation in z axis (radians)" << endl;
	cout << "--bg        specify file with list of backround images to superimpose extracted images onto" << endl;
	cout << "--no-shifts don't generate shifted calibration outputs" << endl;
}


vector<string> Arguments(int argc, char *argv[])
{
    size_t temp_pos;
    int    temp_int;

    vector<string> vid_names;
    vid_names.push_back("");
    if (argc < 2)
    {
        usage(argv);
    }
    else if (argc == 2)
    {
        vid_names[0] = argv[1];
    }
    else
    {
        for (int i = 0; i < argc; i++)
        {
            if (strncmp(argv[i], "-r", 2) == 0)
            {
                try
                {
                    stoi(argv[i + 1], &temp_pos, 16);
                    if (temp_pos != 6)
                    {
                        cout << "Wrong number of hex digits for param -r!" << endl;
                        break;
                    }
                    stoi(argv[i + 2], &temp_pos, 16);
                    if (temp_pos != 6)
                    {
                        cout << "Wrong number of hex digits for param -r!" << endl;
                        break;
                    }
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                temp_int  = stoi(argv[i + 1], &temp_pos, 16);
                g_h_min   = temp_int / 65536;
                temp_int -= g_h_min * 65536;
                g_s_min   = temp_int / 256;
                temp_int -= g_s_min * 256;
                g_v_min   = temp_int;
                temp_int  = stoi(argv[i + 2], &temp_pos, 16);
                g_h_max   = temp_int / 65536;
                temp_int -= g_h_min * 65536;
                g_s_max   = temp_int / 256;
                temp_int -= g_s_min * 256;
                g_v_max   = temp_int;
                i        += 2;
            }
            else if (strncmp(argv[i], "-f", 2) == 0)
            {
                try
                {
					g_num_frames = stoi(argv[i + 1]);
                    if (g_num_frames < 1)
                    {
                        cout << "Must get at least one frame per file!" << endl;
                        break;
                    }
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                i           += 1;
            }
            else if (strncmp(argv[i], "--min", 4) == 0)
            {
                try
                {
					g_min_resize = stoi(argv[i + 1]);
                    if (g_min_resize < 0)
                    {
                        cout << "Cannot resize below 0%!" << endl;
                        break;
                    }
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                i++;
            }
            else if (strncmp(argv[i], "--maxxrot", 9) == 0)
            {
                try
                {
					g_maxrot.x = stod(argv[i + 1]);
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                i++;
            }
            else if (strncmp(argv[i], "--maxyrot", 9) == 0)
            {
                try
                {
                    g_maxrot.y = stod(argv[i + 1]);
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                i++;
            }
            else if (strncmp(argv[i], "--maxzrot", 9) == 0)
            {
                try
                {
                    g_maxrot.z = stod(argv[i + 1]);
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                i++;
            }
            else if (strncmp(argv[i], "--max", 4) == 0)
            {
                try
                {
					g_max_resize = stoi(argv[i + 1]);
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                i++;
            }
            else if (strncmp(argv[i], "--no-shifts", 11) == 0)
            {
                g_do_shifts = false;
            }
            else if (strncmp(argv[i], "-i", 2) == 0)
            {
                try
                {
					g_files_per = stoi(argv[i + 1]);
                    if (g_files_per < 1)
                    {
                        cout << "Must output at least 1 file per frame!" << endl;
                        break;
                    }
                }
                catch (...)
                {
                    usage(argv);
                    break;
                }
                i++;
            }
            else if (strncmp(argv[i], "-o", 2) == 0)
            {
                g_outputdir = argv[i + 1];
                i++;
            }
            else if (strncmp(argv[i], "--bg", 4) == 0)
            {
                g_bgfile = argv[i + 1];
                i++;
            }
            else if (argv[i] != argv[0])
            {
                for ( ; i < argc; i++)
                {
                    if (vid_names[0] == "")
                    {
                        vid_names[0] = argv[i];
                    }
                    else
                    {
                        vid_names.push_back(argv[i]);
                    }
                }
            }
        }
    }
    return vid_names;
}


typedef pair<float, int> Blur_Entry;
void readVideoFrames(const string &vidName, int &frameCounter, vector<Blur_Entry> &lblur)
{
	VideoCapture frameVideo(vidName);

	lblur.clear();
	frameCounter = 0;
	if (!frameVideo.isOpened())
	{
		return;
	}

	Mat frame;
	Mat hsvInput;

	Mat temp;
	Mat tempm;
	Mat gframe;
	Mat variancem;

#ifndef DEBUG_FOO
	// Grab a list of frames which have an identifiable
	// object in them.  For each frame, compute a
	// blur score indicating how clear each frame is
	for (frameCounter = 0; frameVideo.read(frame); frameCounter += 1)
	{
		cvtColor(frame, hsvInput, CV_BGR2HSV);
		Rect bounding_rect;
		if (FindRect(hsvInput, Scalar(g_h_min, g_s_min, g_v_min), Scalar(g_h_max, g_s_max, g_v_max), bounding_rect))
		{
			cvtColor(frame, gframe, CV_BGR2GRAY);
			Laplacian(gframe, temp, CV_8UC1);
			meanStdDev(temp, tempm, variancem);
			float variance = pow(variancem.at<Scalar>(0, 0)[0], 2);
			lblur.push_back(Blur_Entry(variance, frameCounter));
		}
	}
#else
	frameCounter = 1;
	lblur.push_back(Blur_Entry(1,187));
#endif
	sort(lblur.begin(), lblur.end(), greater<Blur_Entry>());
	cout << "Read " << lblur.size() << " valid frames from video of " << frameCounter << " total" << endl;
}


int main(int argc, char *argv[])
{
    vector<string> vid_names = Arguments(argc, argv);
    if (vid_names[0] == "")
    {
        cout << "Invalid program syntax!" << endl;
        return 0;
    }
#ifdef DEBUG
    namedWindow("RangeControl", WINDOW_AUTOSIZE);

    createTrackbar("HueMin", "RangeControl", &g_h_min, 255);
    createTrackbar("HueMax", "RangeControl", &g_h_max, 255);

    createTrackbar("SatMin", "RangeControl", &g_s_min, 255);
    createTrackbar("SatMax", "RangeControl", &g_s_max, 255);

    createTrackbar("ValMin", "RangeControl", &g_v_min, 255);
    createTrackbar("ValMax", "RangeControl", &g_v_max, 255);
#endif

	RNG rng(time(NULL));

	// Middle of chroma-key range
    Vec3b mid((g_h_min + g_h_max) / 2, (g_s_min + g_s_max) / 2, (g_v_min + g_v_max) / 2);

	// Load in a list of images to use as backgrounds for
	// chroma-keying 
	// If none are present, RandomSubImage returns an
	// image filled with random pixel values
	vector<string> bgFileList;
	if (g_bgfile.length())
	{
		ifstream bgfile(g_bgfile);
		string bgfilename;
		while (getline(bgfile, bgfilename))
		{
			bgFileList.push_back(bgfilename);
		}
		bgfile.close();
	}
	RandomSubImage rsi(rng, bgFileList);

	// Create output directories
	if (mkdir(g_outputdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
	{
		if (errno != EEXIST)
		{
			cerr << "Could not create " << g_outputdir.c_str() << ":";
			perror("");
			return -1;
		}
	}

	if (g_do_shifts && !createShiftDirs(g_outputdir + "/shifts"))
		return -1;

	// Iterate through each input video
    for (auto vidName = vid_names.cbegin(); vidName != vid_names.cend(); ++vidName)
    {
        cout << *vidName << endl;

        Mat frame;
		Mat hsvframeIn; 
		Mat mask;
        Mat hsvframe; // hsv version of input frame padded with empty pixels
		Mat objMask;  // mask padded to size of hsvframe
		Mat objMaskInv;
		Mat bgImg;    // random background image to superimpose each input onto 
		Mat chromaImg; // combined input plus bg
		Mat rotImg;  // randomly rotated input
		Mat rotMask; // and mask
		Mat hsv_final; // final processed data in HSV format
		Mat splitMat[3];
		Rect bounding_rect;

        int   frame_counter;

		// Grab an array of frames sorted by how clear they are
        vector<Blur_Entry> lblur;
		readVideoFrames(*vidName, frame_counter, lblur);
		if (lblur.empty())
        {
            cout << "Capture not open; invalid video " << *vidName << endl;
            continue;
        }

        VideoCapture frame_video(*vidName);
        int          frame_count = 0;
        vector<bool> frame_used(frame_counter);
        const int    frame_range = lblur.size()/100;      // Try to space frames out by this many unused frames
        for (auto it = lblur.begin(); (frame_count < g_num_frames) && (it != lblur.end()); ++it)
        {
            // Check to see that we haven't used a frame close to this one
            // already - hopefully this will give some variety in the frames
            // which are used
            int  this_frame      = it->second;
            bool frame_too_close = false;
            for (int j = max(this_frame - frame_range + 1, 0); !frame_too_close && (j < min((int)frame_used.size(), this_frame + frame_range)); j++)
            {
                if (frame_used[j])
                {
                    frame_too_close = true;
                }
            }

            if (frame_too_close)
            {
                continue;
            }

            frame_used[this_frame] = true;

            frame_video.set(CV_CAP_PROP_POS_FRAMES, this_frame);
            frame_video >> frame;
            cvtColor(frame, hsvframeIn, CV_BGR2HSV);

#ifdef DEBUG
			imshow("Frame at read", frame);
			imshow("HSV Frame at read", hsvframeIn);
#endif

			// Get a mask image. Pixels for the object in question
			// will be set to 255, others to 0
			if (!getMask(hsvframeIn, Scalar(g_h_min, g_s_min, g_v_min), Scalar(g_h_max, g_s_max, g_v_max), mask, bounding_rect))
			{
				continue;
			}
			bounding_rect = AdjustRect(bounding_rect, 1.0);

			// Expand the input image, padding it with pixels
			// set to the chroma key color.  Do the same for the
			// mask, but fill with 0 (non-object) pixels and 
			// update the bounding_rect coords to match.
			// This adds extra
			// border for cases where grabbing a larger rect
			// around the object would have gone off the edge
			// of the original input image
			const int expand = max(hsvframeIn.rows, hsvframeIn.cols) * 2;
			copyMakeBorder(hsvframeIn, hsvframe, expand, expand, expand, expand, BORDER_CONSTANT, Scalar(mid));
			copyMakeBorder(mask, objMask, expand, expand, expand, expand, BORDER_CONSTANT, Scalar(0));
			bounding_rect = Rect(expand+bounding_rect.x, expand+bounding_rect.y, bounding_rect.width, bounding_rect.height);
#ifdef DEBUG
			imshow("Original size mask", mask);
			imshow("Resized HSV image", hsvframe);
			imshow("Resized mask", objMask);
#endif
			// This shouldn't happen
			Rect frame_rect(Point(0,0), hsvframe.size());
			if ((bounding_rect & frame_rect) != bounding_rect)
			{
				cout << "Rectangle " << bounding_rect << "out of bounds of frame " << hsvframe.size() << endl;
				continue;
			}

			// Fill non-mask pixels with midpoint
			// of chroma-key color.  This will set pixels 
			// at the edge of the image to the chroma key
			// color even if we happen to shoot video where
			// the egde of the image goes a bit beyond the
			// green screen
			Mat chromaFill(hsvframe.size(), CV_8UC3);
			chromaFill = Scalar(mid);
			bitwise_not(objMask, objMaskInv);
			chromaFill.copyTo(hsvframe, objMaskInv);

#ifdef DEBUG
            imshow("objMask returned from getMask", objMask);
            imshow("HSV frame after fill with mid", hsvframe);
#endif

			// Randomly adjust the hue - this will hopefully
			// simulate an object reflecting colored light
            hsvframe.convertTo(hsvframe, CV_32FC3);
            for (int hueAdjust = 0; hueAdjust <= 160; hueAdjust += 30)
            {
				int rndHueAdjust = hueAdjust + rng.uniform(-10,10);
                for (int l = 0; l < hsvframe.rows; l++)
                {
					const uchar *mask = objMask.ptr<uchar>(l);
					Vec3f *pt = hsvframe.ptr<Vec3f>(l);
                    for (int m = 0; m < hsvframe.cols; m++)
                    {
						if (mask[m])
						{
							float val = pt[m][0];
							val += rndHueAdjust;
							if (val >= 180.)
								val = val - 180.0;
							else if (val < 0.)
								val = val + 180.0;
							pt[m][0] = val;
						}
                    }
                }
				// Make sure this value is positive when
				// used to print filename
				if (rndHueAdjust < 0)
					rndHueAdjust += 180;

				// Add gaussian noise to the image
				// S and V channels.
				// TODO : should this generate a different
				// set of noise values for S and V?
                Mat noise = Mat::zeros(hsvframe.size(), CV_32F);
                randn(noise, 0.0, g_noise);
                split(hsvframe, splitMat);

                for (int i = 1; i <= 2; i++)
                {
                    double min, max;
                    minMaxLoc(splitMat[i], &min, &max, NULL, NULL, objMask);
                    add(splitMat[i], noise, splitMat[i], objMask);
                    normalize(splitMat[i], splitMat[i], min, max, NORM_MINMAX, -1, objMask);
                }

				// Convert back to 8 bit BGR value
				// to prepare for final steps and
				// then saving as a normal image file
                merge(splitMat, 3, hsv_final);
                hsv_final.convertTo(hsv_final, CV_8UC3);
                cvtColor(hsv_final, frame, CV_HSV2BGR);
#ifdef DEBUG
                imshow("Final HSV", hsv_final);
                imshow("Final RGB", frame);
                waitKey(0);
#endif
				// Generate shifted inputs for training
				// calibration networks
				if (g_do_shifts)
				{
					stringstream shift_fn;
					shift_fn << g_outputdir << "/" + Behead(*vidName) << "_" << setw(5) << setfill('0') << this_frame;
					shift_fn << "_" << setw(4) << bounding_rect.x;
					shift_fn << "_" << setw(4) << bounding_rect.y;
					shift_fn << "_" << setw(4) << bounding_rect.width;
					shift_fn << "_" << setw(4) << bounding_rect.height;
					shift_fn << "_" << setw(3) << rndHueAdjust;
					shift_fn << ".png";
					doShifts(frame(bounding_rect), objMask(bounding_rect), 
							 rng, rsi, g_maxrot, 4, 
							 g_outputdir + "/shifts", shift_fn.str());
				}

				// Generate g_files_per randomly resized and rotated
				// copies of this input frame, each superimposed onto
				// a random background image (or random RGB values if
				// no list of background images were specified)
				int fail_count = 0;
                for (int i = 0; (i < g_files_per) && (fail_count < 100); )
                {
					double scale_up = rng.uniform((double)g_min_resize, (double)g_max_resize);
					Rect final_rect;
					// This will be true if the rescaled rect fits inside the 
					// input frame
					// Since earlier code expands the input frame to 5x the 
					// input size this should never fail
                    if (RescaleRect(bounding_rect, final_rect, frame.size(), scale_up))
                    {
						// Probably faster to do this outside loop - make
						// 2 versions of code for GPU vs CPU?
						if (cuda::getCudaEnabledDeviceCount() > 0)
						{
							GpuMat f(frame(final_rect));
							GpuMat m(objMask(final_rect));
							GpuMat rI;
							GpuMat rM;
							rotateImageAndMask(f, m, Scalar(frame(final_rect).at<Vec3b>(0,0)), g_maxrot, rng, rI, rM);
							bgImg = rsi.get((double)frame.cols / frame.rows, 0.05);
							GpuMat cI = doChromaKey(rI, GpuMat(bgImg), rM);
							GpuMat out;
							cuda::resize(cI, out, Size(48,48));
							out.download(chromaImg);
						}
						else
						{
							rotateImageAndMask(frame(final_rect), objMask(final_rect), Scalar(frame(final_rect).at<Vec3b>(0,0)), g_maxrot, rng, rotImg, rotMask);
#ifdef DEBUG
							imshow("FinalRGB(final_rect)", frame(final_rect));
							imshow("Mask(final_rect)", objMask(final_rect));
							imshow("rotImg", rotImg);
							imshow("rotMask", rotMask);
							waitKey(0);
#endif
							bgImg = rsi.get((double)frame.cols / frame.rows, 0.05);
							chromaImg = doChromaKey(rotImg, bgImg, rotMask);
							resize(chromaImg, chromaImg, Size(48,48));
						}

                        stringstream write_name;
                        write_name << g_outputdir << "/" + Behead(*vidName) << "_" << setw(5) << setfill('0') << this_frame;
                        write_name << "_" << setw(4) << final_rect.x;
                        write_name << "_" << setw(4) << final_rect.y;
                        write_name << "_" << setw(4) << final_rect.width;
                        write_name << "_" << setw(4) << final_rect.height;
                        write_name << "_" << setw(3) << rndHueAdjust;
                        write_name << "_" << setw(3) << i;
                        write_name << ".png";
                        if (imwrite(write_name.str().c_str(), chromaImg) == false)
						{
							cout << "Error! Could not write file "<<  write_name.str() << endl;
							fail_count += 1;
						}
						else
						{
							i++;
							fail_count = 0;
						}
                    }
					else
						fail_count += 1;
                }
            }
            frame_count += 1;
        }
    }
	// Display range and midpoint of chroma key values used
	// to mask off object.  Used to be needed for old image
	// generation toolchain, not so much anymore now that 
	// chroma-keying and rotation were combined into this
	// program.
    cout << "0x" << IntToHex((g_h_min + g_h_max) / 2) << IntToHex((g_s_min + g_s_max) / 2) << IntToHex((g_v_min + g_v_max) / 2);
    cout << " 0x" << IntToHex((g_h_min + g_h_max) / 2 - g_h_min) << IntToHex((g_s_min + g_s_max) / 2 - g_s_min) << IntToHex((g_v_min + g_v_max) / 2 - g_v_min) << endl;
    return 0;
}
