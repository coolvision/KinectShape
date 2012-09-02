/*
 * Voxels.h
 *
 *  Created on: 2012
 *      Author: sk
 */

#pragma once

class VoxelVolume {
public:
	// voxel box dimensions
	int side_n;
	ofPoint size; // voxel size in mm
	ofPoint min; // min corner in the mm space

	// truncated surface distance data for each voxel
	float *data;
	size_t bytes_n;
	int array_size;

	// data for voxel weights
	unsigned char *w_data;
	size_t w_bytes_n;
};
