KinectShape
===========

Implementation of "KinectFusion" 3d shape reconstruction method, described in:

_"Richard A. Newcombe, Shahram Izadi, Otmar Hilliges, David Molyneaux, David Kim, Andrew J. Davison, Pushmeet Kohli, Jamie Shotton, Steve Hodges, Andrew W. Fitzgibbon: KinectFusion: Real-time dense surface mapping and tracking. ISMAR 2011: 127-136"_

For detailed description of this project, see the blog post: http://k10v.com/2012/09/02/18/

Released under the MIT license.

Build instructions
------------------
All underlying libraries and frameworks are cross-platform (Openframeworks, libfreenect, libeigen, CUDA SDK), so the project is supposed to be cross-platform.
But for now, only makefile for linux is maintained, and it is tested only on Ubuntu 12.04.
Requirements:
- Openframewokls 0071 with dependencies installed
- CUDA 4.1 setup: SDK, toolkit, dev driver; nvcc is supposed to be in $PATH; path to SDK and toolkit should be specified in config.make
- libusb-dev intalled
- libeigen3-dev intalled
- libfreenect-dev installed
- addons from ./all_addons directory pasted into OF_ROOT/addons (2do: should properly fork modified addons and add them as submodules)
- OF_ROOT set to Openframeworks path on config.make
- launch as root user, or use instructions on how to run OpenKinect without root privilegies (http://openkinect.org/wiki/Getting_Started#Ubuntu_Manual_Install)