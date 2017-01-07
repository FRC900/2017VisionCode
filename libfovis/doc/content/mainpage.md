FOVIS {#mainpage}
===============

# Introduction

Fovis is a visual odometry library that estimates the 3D motion of a camera
using a source of depth information for each pixel.  Its original
implementation is described in the paper:

 Visual Odometry and Mapping for Autonomous Flight Using an RGB-D Camera. <br/>
 _Albert S. Huang, Abraham Bachrach, Peter Henry, Michael Krainin, Daniel Maturana, Dieter Fox, and Nicholas Roy_. <br/>
 Int. Symposium on Robotics Research (ISRR), Flagstaff, Arizona, USA, Aug. 2011
 ([PDF](http://people.csail.mit.edu/albert/pubs/2011-huang-isrr.pdf)).

# Quick links
 - [Downloads](https://github.com/fovis/fovis/releases)
 - [GitHub site](https://github.com/fovis/fovis)

# Build requirements

Fovis is intended to be relatively portable.  There is requirements
for building and using the software are:
- [Eigen 3](http://eigen.tuxfamily.org)
- [CMake](http://www.cmake.org)

Fovis was developed on Ubuntu, but may work with other platforms.

# Build instructions

For system-wide installation:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ sudo make install
    $ sudo ldconfig

This usually installs Fovis to /usr/local or something like that.

For use in the source directory:

    $ mkdir build
    $ make

## Intel SSE
By default, Fovis is configured to use Intel SSE2 and SSE3 instructions.  You
can disable this option (e.g., if you're targeting a platform without those
instructions) by using the CMake option USE_SSE:

    $ cd build
    $ cmake .. -DUSE_SSE=OFF
    $ make

## Building documentation
Fovis is documented using [Doxygen](http://www.doxygen.org).  To build the documentation:

    $ cd doc
    $ doxygen

Following that, open up doc/html/index.html in your web browser.

# Usage requirements

For portability reasons, the actual library itself is sensor agnostic and
provides no data acquisition capabilities.  To use fovis, your program must
acquire data on its own and pass it through to the fovis API.  Some examples
are provided with the source code.

Effective use of fovis for visual odometry requires the following:
- A source of 8-bit grayscale camera images.
- A camera calibration for the images that provides an accurate mapping between
image pixel coordinates (u, v) and 3D rays (X, Y, Z) in the camera's Cartesian
coordinate frame.
- A depth source for each image.  A depth source must be able to provide a
metric depth estimate for as many pixels in the camera image as possible.

Fovis provides built-in support for the following types of depth sources:
- An RGB-D camera such as the Microsoft Kinect.
- Calibrated stereo cameras.

You can also create your own depth sources using the Fovis API and adapt it to
other sensor types.

# Getting started

The best way to get started is to look through the examples provided with the
source code in the examples/ directory.

Next, look through the Fovis C++ API.  The primary class of interest
is \ref fovis::VisualOdometry.

# License

fovis is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

fovis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

A copy of the GNU General Public License is provided with the fovis source
code.
