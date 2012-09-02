/*
 * VoxelsCUDA.h
 *
 *  Created on: 2012
 *      Author: sk
 */

#pragma once

#include <cuda.h>
#include <vector_types.h>

class VoxelVolumeCUDA {
public:

	int side_n;
	int side_n2;

	float3 size;
	float3 min;

	// truncated surface distance data for each voxel
	float *data;
	size_t bytes_n;
	int array_size;

	// data for voxel weights
	unsigned char *w_data;
	size_t w_bytes_n;
};
