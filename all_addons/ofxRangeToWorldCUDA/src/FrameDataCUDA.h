/*
 * FrameDataCUDA.h
 *
 *  Created on: 2012
 *      Author: sk
 */

#pragma once

class FrameDataCUDA {
public:
	float *depth;
	float *points;
	float *normals;

	size_t depth_bn;
	size_t points_bn;

	int width;
	int height;
	int step;
};

class CameraOptions {
public:
	float *t; // transform matrix
	float *it; // inverse transform
	float ref_pix_size;
	float ref_distance;

	// 3d volume filtering
	float min[3];
	float max[3];
};
