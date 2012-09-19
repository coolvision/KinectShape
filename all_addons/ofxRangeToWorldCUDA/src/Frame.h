/*
 * Frame.h
 *
 *  Created on: 2012
 *      Author: sk
 */

#pragma once

#include "ofMain.h"
#include "ofxKinect.h"
#include "FrameData.h"
#include "FrameDataCUDA.h"

class Frame {
public:

	// world transform corresponding to this data
	ofMatrix4x4 t;
	ofMatrix4x4 it;

	// data size
	size_t depth_bn;
	size_t points_bn;

	// data on the CPU
	FrameData host;

	FrameDataCUDA dev;

	// visualization
	ofMesh mesh;
	ofImage depth_image;
	ofImage color_image;

	void init(int width, int height);
	void setFromPixels(float *pixels);

	void allocateHost();
	void releaseHost();

	void allocateDevice();
	void releaseDevice();

	void meshFromPoints(bool draw_normals, ofxKinect *kinect);
	void drawMesh();
	void drawNormals();
};
