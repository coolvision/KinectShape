/*
 * Utils.cpp
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "Utils.h"

void getPoint(ofPoint *p, float *points, int x, int y, int w) {

	int index = (y * w + x) * 3;
	p->x = points[index];
	p->y = points[index + 1];
	p->z = points[index + 2];
}

void setPoint(float *points, int x, int y, int w,
		float fx, float fy, float fz) {

	int index = (y * w + x) * 3;
	points[index] = fx;
	points[index + 1] = fy;
	points[index + 2] = fz;
}
