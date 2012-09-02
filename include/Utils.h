/*
 * Utils.h
 *
 *  Created on: 2012
 *      Author: sk
 */

#pragma once

#include "ofMain.h"
#include "ofxKinect.h"

void getPoint(ofPoint *p, float *points, int x, int y, int w);

void setPoint(float *points, int x, int y, int w,
		float fx, float fy, float fz);
