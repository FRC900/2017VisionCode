/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
 * cvsamples.cpp
 *
 * support functions for training and test samples creation.
 */
#include "cvhaartraining.h"
#include "_cvhaartraining.h"

#include "cv.h"
#include "highgui.h"

/* Warps source into destination by a perspective transform */
static void WarpPerspective( const cv::Mat &src, cv::Mat &dst, double quad[4][2] )
{
	cv::Mat output;
    // Input Quadilateral or Image plane coordinates
	cv::Point2f inputQuad[4]; 
    // Output Quadilateral or World plane coordinates
	cv::Point2f outputQuad[4];

    for( int i = 0; i < 4; ++i )
		outputQuad[i] = cv::Point2f(quad[i][0], quad[i][1]);

	inputQuad[0] = cv::Point2f(0, 0);
	inputQuad[1] = cv::Point2f(src.cols - 1, 0);
	inputQuad[2] = cv::Point2f(src.cols - 1, src.rows - 1);
	inputQuad[3] = cv::Point2f(0, src.rows - 1);

	cv::Mat lambda = cv::getPerspectiveTransform( inputQuad, outputQuad );

	cv::warpPerspective (src, dst, lambda, dst.size());
}


static
void icvRandomQuad( int width, int height, double quad[4][2],
                    double maxxangle,
                    double maxyangle,
                    double maxzangle )
{
    double distfactor = 3.0;
    double distfactor2 = 1.0;

    double halfw, halfh;
    int i;

    double rotVectData[3];
    double vectData[3];
    double rotMatData[9];

    CvMat rotVect;
    CvMat rotMat;
    CvMat vect;

    double d;

    rotVect = cvMat( 3, 1, CV_64FC1, &rotVectData[0] );
    rotMat = cvMat( 3, 3, CV_64FC1, &rotMatData[0] );
    vect = cvMat( 3, 1, CV_64FC1, &vectData[0] );

    rotVectData[0] = maxxangle * (2.0 * rand() / RAND_MAX - 1.0);
    rotVectData[1] = ( maxyangle - fabs( rotVectData[0] ) )
        * (2.0 * rand() / RAND_MAX - 1.0);
    rotVectData[2] = maxzangle * (2.0 * rand() / RAND_MAX - 1.0);
    d = (distfactor + distfactor2 * (2.0 * rand() / RAND_MAX - 1.0)) * width;

#if 0
    rotVectData[0] = maxxangle;
    rotVectData[1] = maxyangle;
    rotVectData[2] = maxzangle;

    d = distfactor * width;
#endif

    cvRodrigues2( &rotVect, &rotMat );

    halfw = 0.5 * width;
    halfh = 0.5 * height;

    quad[0][0] = -halfw;
    quad[0][1] = -halfh;
    quad[1][0] =  halfw;
    quad[1][1] = -halfh;
    quad[2][0] =  halfw;
    quad[2][1] =  halfh;
    quad[3][0] = -halfw;
    quad[3][1] =  halfh;

    for( i = 0; i < 4; i++ )
    {
        rotVectData[0] = quad[i][0];
        rotVectData[1] = quad[i][1];
        rotVectData[2] = 0.0;
        cvMatMulAdd( &rotMat, &rotVect, 0, &vect );
        quad[i][0] = vectData[0] * d / (d + vectData[2]) + halfw;
        quad[i][1] = vectData[1] * d / (d + vectData[2]) + halfh;

        /*
        quad[i][0] += halfw;
        quad[i][1] += halfh;
        */
    }
}


int icvStartSampleDistortion( const char* imgfilename, int bgcolor, int bgthreshold,
                              CvSampleDistortionData* data, bool grayscale, bool hsv )
{
	if (grayscale && hsv)
	   return 0;
    memset( data, 0, sizeof( *data ) );
    data->src = cvLoadImage( imgfilename, grayscale ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR );
    if( data->src != NULL && data->src->depth == IPL_DEPTH_8U )
    {
        int r, c, n, m; // row, col, channel num
        uchar* psrc;    // pointer to current source image pixel(s)
        uchar* perode;  // pointer to erode and dialate pixels
        uchar* pdilate;
        uchar dd, de;
		std::vector<int> bgcolors;    
		std::vector<int> bgthresholds;
		cv::Mat hsvImage;

		/* convert bgcolor/bgthreshold into array of component BGR values */
		for ( n = 0; n < data->src->nChannels; n++ )
		{
			/* bgcolor / bgthreshold are 3 8-bit numbers in RGB / HSV format
			 * pixels are stored in BGR or HSV. Make the bgcolors/bgthresholds
			 * arrays store the values in that order too to make indexing easy */
		    bgcolors.push_back    (((unsigned)bgcolor     >> (n * 8)) & 0xff);
		    bgthresholds.push_back(((unsigned)bgthreshold >> (n * 8)) & 0xff);
		}
		
		if (hsv)
		{
		   // Above will create VSH - swap it to match HSV pixel alignment
		   std::swap(bgcolors[0], bgcolors[2]);
		   std::swap(bgthresholds[0], bgthresholds[2]);

		   cv::cvtColor( cv::Mat(data->src), hsvImage, CV_BGR2HSV );
		}

        data->dx = data->src->width / 2;
        data->dy = data->src->height / 2;
        data->bgcolor = bgcolor;

		data->mask = new cv::Mat(data->src->height, data->src->width, CV_8UC1);

        /* make mask image */
        for( r = 0; r < data->mask->rows; r++ )
        {
            for( c = 0; c < data->mask->cols; c++ )
            {
				if (hsv)
					psrc = (uchar*) hsvImage.data + r * hsvImage.step + c * hsvImage.elemSize();
				else
					psrc = (uchar*) (data->src->imageData + r * data->src->widthStep)
							 + c * data->src->nChannels;

				/* Assume sample will be masked off. If any of the channels
				   falls out of range of the mask for that channel, update
				   the pixel to be unmasked and bail out of the loop */
				data->mask->at<uchar>(r,c) = 0;
				for ( n = 0; n < data->src->nChannels; n++ )
				{
		 			if( !(bgcolors[n] - bgthresholds[n] <= (int) psrc[n] &&
	     				(int) psrc[n] <= bgcolors[n] + bgthresholds[n] ) )
	   				{
						data->mask->at<uchar>(r,c) = 255;
	       				break;
	   				}
                }
            }
        }

		// Increase the size of the mask by a small amount
		// to remove a small edge of chroma-key color that
		// seems to surround the images we're working with
		if (hsv)
		   cv::erode( *data->mask, *data->mask, cv::Mat(), cv::Point(-1,-1), 3 ); 

        /* extend borders of source image */
		// TODO this doesn't work for HSV images - we need to compare e&d
		// versions of the HSV image with the thresholds but use e&d version
		// of the original source to extend the borders
        data->erode = cvCloneImage( data->src );
        cvErode( data->src, data->erode, 0, 1 );

        data->dilate = cvCloneImage( data->src );
        cvDilate( data->src, data->dilate, 0, 1 );
        for( r = 0; r < data->mask->rows; r++ )
        {
            for( c = 0; c < data->mask->cols; c++ )
            {
                if( data->mask->at<uchar>(r,c) == 0 )
                {
                    psrc = ( (uchar*) (data->src->imageData + r * data->src->widthStep)
                           + c * data->src->nChannels );
                    perode =
                        ( (uchar*) (data->erode->imageData + r * data->erode->widthStep)
                                + c * data->erode->nChannels );
                    pdilate =
                        ( (uchar*)(data->dilate->imageData + r * data->dilate->widthStep)
                                + c * data->dilate->nChannels );
					for ( n = 0; n < data->src->nChannels; n++ )
					{ 
						de = (uchar)(bgcolors[n] - perode[n]);
						dd = (uchar)(pdilate[n] - bgcolors[n]);
						if( de >= dd && de > bgthresholds[n] )
						{
							for ( m = 0; m < data->src->nChannels; m++ )
							   psrc[m] = perode[m];
						}
						if( dd > de && dd > bgthresholds[n] )
						{
							for ( m = 0; m < data->src->nChannels; m++ )
							   psrc[m] = pdilate[m];
						}
					}
                }
            }
        }


        data->img     = new cv::Mat( data->src->height + 2 * data->dy,
									 data->src->width  + 2 * data->dx,
							         CV_8UC(data->src->nChannels) );
        data->maskimg = new cv::Mat( data->src->height + 2 * data->dy, 
									 data->src->width + 2 * data->dx,
									 CV_8UC1);

        return 1;
    }
    return 0;
}


void icvPlaceDistortedSample( CvArr* background,
                              int inverse, int maxintensitydev,
                              double maxxangle, double maxyangle, double maxzangle,
                              int inscribe, double maxshiftf, double maxscalef,
                              CvSampleDistortionData* data )
{
    double quad[4][2];
    int r, c;
    uchar* pimg;
    uchar* pbg;
    int forecolordev;
    float scale;
    CvMat  stub;
    CvMat* bgimg;

	cv::Rect cr;
	cv::Rect roi;

    double xshift, yshift, randscale;

    icvRandomQuad( data->src->width, data->src->height, quad,
                   maxxangle, maxyangle, maxzangle );
    quad[0][0] += (double) data->dx;
    quad[0][1] += (double) data->dy;
    quad[1][0] += (double) data->dx;
    quad[1][1] += (double) data->dy;
    quad[2][0] += (double) data->dx;
    quad[2][1] += (double) data->dy;
    quad[3][0] += (double) data->dx;
    quad[3][1] += (double) data->dy;
	
	if (data->src->nChannels == 1)
	{
		data->img->setTo( cv::Scalar( data->bgcolor ) );
	}
	else
	{
		// Check HSV vs RGB here
		data->img->setTo( cv::Scalar( (data->bgcolor >> 16) & 0x000000FF ,
		                              (data->bgcolor >>  8) & 0x000000FF ,
		                               data->bgcolor        & 0x000000FF ) );
	}
	WarpPerspective( data->src, *data->img, quad );

    data->maskimg->setTo(cv::Scalar( 0 ));

    WarpPerspective( *data->mask, *data->maskimg, quad );

	cv::GaussianBlur( *data->maskimg, *data->maskimg, cv::Size(3, 3), 1.0 );

    bgimg = cvGetMat( background, &stub );

    cr.x = data->dx;
    cr.y = data->dy;
    cr.width = data->src->width;
    cr.height = data->src->height;

    if( inscribe )
    {
        /* quad's circumscribing rectangle */
        cr.x = (int) MIN( quad[0][0], quad[3][0] );
        cr.y = (int) MIN( quad[0][1], quad[1][1] );
        cr.width  = (int) (MAX( quad[1][0], quad[2][0] ) + 0.5F ) - cr.x;
        cr.height = (int) (MAX( quad[2][1], quad[3][1] ) + 0.5F ) - cr.y;
    }

    xshift = maxshiftf * rand() / RAND_MAX;
    yshift = maxshiftf * rand() / RAND_MAX;

    cr.x -= (int) ( xshift * cr.width  );
    cr.y -= (int) ( yshift * cr.height );
    cr.width  = (int) ((1.0 + maxshiftf) * cr.width );
    cr.height = (int) ((1.0 + maxshiftf) * cr.height);

    randscale = maxscalef * rand() / RAND_MAX;
    cr.x -= (int) ( 0.5 * randscale * cr.width  );
    cr.y -= (int) ( 0.5 * randscale * cr.height );
    cr.width  = (int) ((1.0 + randscale) * cr.width );
    cr.height = (int) ((1.0 + randscale) * cr.height);

    scale = MAX( ((float) cr.width) / bgimg->cols, ((float) cr.height) / bgimg->rows );

    roi.x = (int) (-0.5F * (scale * bgimg->cols - cr.width) + cr.x);
    roi.y = (int) (-0.5F * (scale * bgimg->rows - cr.height) + cr.y);
    roi.width  = (int) (scale * bgimg->cols);
    roi.height = (int) (scale * bgimg->rows);

	cv::Mat img( bgimg->rows, bgimg->cols, CV_8UC(data->src->nChannels ) );
	cv::Mat maskimg( bgimg->rows, bgimg->cols, CV_8UC1 );

	cv::resize( (*data->img)(roi), img, img.size() );
	cv::resize( (*data->maskimg)(roi), maskimg, maskimg.size() );

    forecolordev = (int) (maxintensitydev * (2.0 * rand() / RAND_MAX - 1.0));

	for( r = 0; r < img.rows; r++ )
	{
		for( c = 0; c < img.cols; c++ )
		{
			pimg = (uchar*) img.data + r * img.step + c * img.elemSize();
			pbg  = (uchar*) bgimg->data.ptr + r * bgimg->step + c * img.channels();
			for ( int n = 0; n < img.channels(); n++ )
			{
				uchar chartmp = (uchar) MAX( 0, MIN( 255, forecolordev + pimg[n] ) );
				uchar alpha = maskimg.at<uchar>(r,c);
				if( inverse )
				{
					chartmp ^= 0xFF;
				}
				pbg[n] = (uchar) ( ( chartmp * alpha + (255 - alpha) * pbg[n] ) / 255);
			}
		}
	}
}


void icvEndSampleDistortion( CvSampleDistortionData* data )
{
    if( data->src )
    {
        cvReleaseImage( &data->src );
    }
    if( data->mask )
    {
        delete data->mask;
    }
    if( data->erode )
    {
        cvReleaseImage( &data->erode );
    }
    if( data->dilate )
    {
        cvReleaseImage( &data->dilate );
    }
    if( data->img )
    {
        delete data->img;
    }
    if( data->maskimg )
    {
        delete data->maskimg;
    }
}

void icvWriteVecHeader( FILE* file, int count, int width, int height )
{
    int vecsize;
    short tmp;

    /* number of samples */
    fwrite( &count, sizeof( count ), 1, file );
    /* vector size */
    vecsize = width * height;
    fwrite( &vecsize, sizeof( vecsize ), 1, file );
    /* min/max values */
    tmp = 0;
    fwrite( &tmp, sizeof( tmp ), 1, file );
    fwrite( &tmp, sizeof( tmp ), 1, file );
}

void icvWriteVecSample( FILE* file, CvArr* sample )
{
    CvMat* mat, stub;
    int r, c;
    short tmp;
    uchar chartmp;

    mat = cvGetMat( sample, &stub );
    chartmp = 0;
    fwrite( &chartmp, sizeof( chartmp ), 1, file );
    for( r = 0; r < mat->rows; r++ )
    {
        for( c = 0; c < mat->cols; c++ )
        {
            tmp = (short) (CV_MAT_ELEM( *mat, uchar, r, c ));
            fwrite( &tmp, sizeof( tmp ), 1, file );
        }
    }
}

int cvCreateTrainingSamplesFromInfo( const char* infoname, const char* vecfilename,
                                     int num,
                                     int showsamples,
                                     int winwidth, int winheight )
{
    char fullname[PATH_MAX];
    char* filename;

    FILE* info;
    FILE* vec;
    IplImage* src=0;
    IplImage* sample;
    int line;
    int error;
    int i;
    int x, y, width, height;
    int total;

    assert( infoname != NULL );
    assert( vecfilename != NULL );

    total = 0;
    if( !icvMkDir( vecfilename ) )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to create directory hierarchy: %s\n", vecfilename );
#endif /* CV_VERBOSE */

        return total;
    }

    info = fopen( infoname, "r" );
    if( info == NULL )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to open file: %s\n", infoname );
#endif /* CV_VERBOSE */

        return total;
    }

    vec = fopen( vecfilename, "wb" );
    if( vec == NULL )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to open file: %s\n", vecfilename );
#endif /* CV_VERBOSE */

        fclose( info );

        return total;
    }

    sample = cvCreateImage( cvSize( winwidth, winheight ), IPL_DEPTH_8U, 1 );

    icvWriteVecHeader( vec, num, sample->width, sample->height );

    if( showsamples )
    {
        cvNamedWindow( "Sample", CV_WINDOW_AUTOSIZE );
    }

    strcpy( fullname, infoname );
    filename = strrchr( fullname, '\\' );
    if( filename == NULL )
    {
        filename = strrchr( fullname, '/' );
    }
    if( filename == NULL )
    {
        filename = fullname;
    }
    else
    {
        filename++;
    }
	char *saved_filename = filename;
	char input_str[1024];

    for( line = 1, error = 0, total = 0; total < num ;line++ )
    {
        int count = 0;
		int str_pos = 0;

		error = fgets(input_str, sizeof(input_str), info) == NULL;
		if ( error ) { break; }
		filename = saved_filename;
        error = ( sscanf( input_str, "%s %d %n", filename, &count, &str_pos ) != 2 );
        if( !error )
        {
			// Handle quoted filenames
			if ((filename[0] == '"') && (filename[strlen(filename) - 1] == '"'))
			{
				memmove(filename, filename + 1, strlen(filename) - 1);
				filename[strlen(filename) - 2] = '\0';
			}
            src = cvLoadImage( fullname, 0 );
            error = ( src == NULL );
            if( error )
            {

#if CV_VERBOSE
                fprintf( stderr, "Unable to open image: %s\n", fullname );
				continue;
#endif /* CV_VERBOSE */

            }
        }
		else
		{
			fprintf (stderr, "Error reading filename from line %d : %d %d %d\n", line, error, total, num);
		}

        for( i = 0; (i < count) && (total < num); i++, total++ )
        {
			int chars_read;
            error = ( sscanf( input_str + str_pos, "%d %d %d %d %n", &x, &y, &width, &height, &chars_read ) != 4 );
            if( error ) 
			{
				fprintf (stderr, "Error reading sample %d from line %d\n", i, line);
				break;
			}
			str_pos += chars_read;
            cvSetImageROI( src, cvRect( x, y, width, height ) );
            cvResize( src, sample, width >= sample->width &&
                      height >= sample->height ? CV_INTER_AREA : CV_INTER_LINEAR );

            if( showsamples )
            {
                cvShowImage( "Sample", sample );
                if( cvWaitKey( 0 ) == 27 )
                {
                    showsamples = 0;
                }
            }
            icvWriteVecSample( vec, sample );
        }

        if( src )
        {
            cvReleaseImage( &src );
        }

        if( error )
        {

#if CV_VERBOSE
            fprintf( stderr, "%s(%d) : parse error\n", infoname, line );
#endif /* CV_VERBOSE */

            break;
        }
    }

    if( sample )
    {
        cvReleaseImage( &sample );
    }

    fclose( vec );
    fclose( info );

    return total;
}


void cvShowVecSamples( const char* filename, int winwidth, int winheight,
                       double scale )
{
    CvVecFile file;
    short tmp;
    int i;
    CvMat* sample;

    tmp = 0;
    file.input = fopen( filename, "rb" );

    if( file.input != NULL )
    {
        size_t elements_read1 = fread( &file.count, sizeof( file.count ), 1, file.input );
        size_t elements_read2 = fread( &file.vecsize, sizeof( file.vecsize ), 1, file.input );
        size_t elements_read3 = fread( &tmp, sizeof( tmp ), 1, file.input );
        size_t elements_read4 = fread( &tmp, sizeof( tmp ), 1, file.input );
        CV_Assert(elements_read1 == 1 && elements_read2 == 1 && elements_read3 == 1 && elements_read4 == 1);

        if( file.vecsize != winwidth * winheight )
        {
            int guessed_w = 0;
            int guessed_h = 0;

            fprintf( stderr, "Warning: specified sample width=%d and height=%d "
                "does not correspond to .vec file vector size=%d.\n",
                winwidth, winheight, file.vecsize );
            if( file.vecsize > 0 )
            {
                guessed_w = cvFloor( sqrt( (float) file.vecsize ) );
                if( guessed_w > 0 )
                {
                    guessed_h = file.vecsize / guessed_w;
                }
            }

            if( guessed_w <= 0 || guessed_h <= 0 || guessed_w * guessed_h != file.vecsize)
            {
                fprintf( stderr, "Error: failed to guess sample width and height\n" );
                fclose( file.input );

                return;
            }
            else
            {
                winwidth = guessed_w;
                winheight = guessed_h;
                fprintf( stderr, "Guessed width=%d, guessed height=%d\n",
                    winwidth, winheight );
            }
        }

        if( !feof( file.input ) && scale > 0 )
        {
            CvMat* scaled_sample = 0;

            file.last = 0;
            file.vector = (short*) cvAlloc( sizeof( *file.vector ) * file.vecsize );
            sample = scaled_sample = cvCreateMat( winheight, winwidth, CV_8UC1 );
            if( scale != 1.0 )
            {
                scaled_sample = cvCreateMat( MAX( 1, cvCeil( scale * winheight ) ),
                                             MAX( 1, cvCeil( scale * winwidth ) ),
                                             CV_8UC1 );
            }
            cvNamedWindow( "Sample", CV_WINDOW_AUTOSIZE );
            for( i = 0; i < file.count; i++ )
            {
                icvGetHaarTraininDataFromVecCallback( sample, &file );
                if( scale != 1.0 ) cvResize( sample, scaled_sample, CV_INTER_LINEAR);
                cvShowImage( "Sample", scaled_sample );
                if( cvWaitKey( 0 ) == 27 ) break;
            }
            if( scaled_sample && scaled_sample != sample ) cvReleaseMat( &scaled_sample );
            cvReleaseMat( &sample );
            cvFree( &file.vector );
        }
        fclose( file.input );
    }
}


/* End of file. */
