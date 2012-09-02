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
