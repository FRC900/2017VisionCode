Fork of OpenCV 2.4.12.2 createsamples utility, modified to handle color images using -pngfnformat flag.

Changes 

Modified meaning of -bgvalue and -bgthresh. These can be 24-bit hex values (e.g. 0xaabbcc) for color operations, formatted as 0xrrggbb or 0xhhssvv for both value and threshold.
Added -pngfnformat to take an [sf]printf format string. This is used to generate mulitple output png files from a single input. The output files will be shifted, modified in intensity and masked based various command line options
Passing in no background image will make the background randomized for color and grayscale inputs. This is only implemented for cases where the output is a .vec file (for grayscale) or for multi-PNG file output (for color ops)
Added -hsvthresh to do thresholding in HSV colorspace instead of the default RGB

This tool is used to process images before they are passed into object detection training code.  Our typical use case is to take a single image and create mulitple modified outputs.  Each output will be rotated a random amount and have a randomly modified intensity. Each image can also have the background masked off so the foreground image can be superimposed on either a different random image or just filled with random pixels.

The command line options we typically use in all cases :

-num <sample count> : the number of output images to make from a single input image
-img <filename> : the input image
-maxidev <idev> : maximum intensity deviation
-maxxangle <radians>, -maxyangle <radians>, -maxzangle <radians> : limits for the random rotation applied to each image.  x moves the top of the image closer or further, y moves the left and right and z spins the image.
-w <int> -h <int> : width and height of the generated output images
-bgvalue <value>, -bgthresh<value> : specify which colors are background to be masked off.  For grayscale, each is just a value from 0-255.  Colors within (val - thresh) to (val + thresh) will be masked off as background.  For 3-channel images, use a 24-bit hex number instead - 0xrrggbb or0xhhssvv depending on the colorspace being used.
-hsvthresh : mask off using HSV color values rather than the default of RGB. 
-bg <filename> : filename contains a list of images to use to replace the background masked off using -bgvalue and -bgthresh.  If this flag isn't present, the masked off area is just filled with random-valued pixels.


For training cascade classifiers, the training code expects input in the form of a .vec file. This is simply a collection of equally-sized grayscale images. The option -vec is used to name the output file created.  The following options will display a .vec file :

-w <width> -h <height> -vec <input vec filename> -show

For training neural nets, we just need a set of .png files.  For this, use the -pngfnformat option. The argument is a format string used to name the output files. For example, "output_%3.3d.png" will create ouput_000.png, output_001.png, and so on.


