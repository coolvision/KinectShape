/*
 * Settings.h
 *
 *  Created on: 2012
 *      Author: sk
 */

#pragma once

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480
#define DEPTH_X_RES 640
#define DEPTH_Y_RES 480

#define SDF_T 0.05f

#define SDF_NONE -FLT_MAX

#define CORRESPONDENCE_BLOCK_SIZE 1024
#define CORRESPONDENCE_BLOCK_X 32
#define CORRESPONDENCE_BLOCK_Y 32

// array for storing normal equations summands
// can store just half of 6x6 matrix: 18 elements
// but for now all elements are moved around
#define AtA_SIZE 36
// right side
#define Atb_SIZE 6

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

class CorrespondenceCUDA {
public:

	// distance and normals thresholds
	float d_t;
	float n_t;

	// normal matrices
	// are split into CUDA blocks
	int blocks_x;
	int blocks_y;
	int blocks_n;

	// calculated in parallel
	float *AtA_dev;
	int AtA_dev_size;
	float *Atb_dev;
	int Atb_dev_size;

	// copied back
	float *AtA_host;
	float *Atb_host;

	// and summed on CPU
	float *AtA_sum;
	float *Atb_sum;

	// return found correspondence point for visualization
	float *points_dev;
	float *points_host;
};

enum RayStopStatus {
	MISS_VOLUME = 0,
	MISS_SURFACE,
	STOP_AT_SURFACE,
	ITERATIONS_EXPIRED,
	NO_MEASUREMENT,
	NEGATIVE_CROSS,
	LOW_WEIGHT,
};
