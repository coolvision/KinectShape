/*
 * FrameData.h
 *
 *  Created on: 2012
 *      Author: sk
 */

#pragma once

class FrameData {
public:
	float *depth;
	float *points;
	float *normals;

	size_t depth_bn;
	size_t points_bn;

	int width;
	int height;
	int step;

	void getPoint(ofPoint *point, int x, int y);
	void getNormal(ofPoint *point, int x, int y);
};
