/*
 * Common.cu
 *
 *  Created on: 2012
 *      Author: sk
 */

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <float.h>

#include <cutil_inline.h>
#include <cutil_math.h>

#include "Settings.h"
#include "VoxelsCUDA.h"

__forceinline__
int divUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ __forceinline__
float3 getPoint(float *p, int x, int y, int w) {
	float3 result;
	int index = (y * w + x) * 3;
	result.x = p[index];
	result.y = p[index + 1];
	result.z = p[index + 2];
	return result;
}

__device__ __forceinline__
void setPoint(float3 v, float *p, int x, int y, int w) {
	int index = (y * w + x) * 3;
	p[index] = v.x;
	p[index + 1] = v.y;
	p[index + 2] = v.z;
}


__device__ __forceinline__
float3 matrixVectorMultiply(const float* t, const float3 &v) {
	float3 result;
	float d = 1.0f / (t[3] * v.x + t[7] * v.y + t[11] * v.z + t[15]);
	result.x = (t[0]*v.x + t[4]*v.y + t[8]*v.z + t[12])*d;
	result.y = (t[1]*v.x + t[5]*v.y + t[9]*v.z + t[13])*d;
	result.z = (t[2]*v.x + t[6]*v.y + t[10]*v.z + t[14])*d;
	return result;
}

// 2do change camera parameters to meters everywhere
// project from the world to the image plane
__device__ __forceinline__
float3 cameraProject(const float ref_pix_size,
		const float ref_distance, const float3* v) {
	float3 result;
	float factor = (2 * ref_pix_size * (*v).z)/(ref_distance);
	result.x = (*v).x / factor + DEPTH_X_RES / 2;
	result.y = (*v).y / factor + DEPTH_Y_RES / 2;
	result.z = (*v).z;
	return result;
}

// from the image to the world
__device__ __forceinline__
float3 cameraBackProject(const float ref_pix_size,
		const float ref_distance, const float3 &v) {

	float3 result;
	float factor = 2 * ref_pix_size * v.z / ref_distance;
	result.x = (v.x - DEPTH_X_RES / 2) * factor;
	result.y = (v.y - DEPTH_Y_RES / 2) * factor;
	result.z = v.z;

	// camera parameters are set for mm coords, transform to meters
	result /= 1000.0f;

	return result;
}

__device__ __forceinline__
float3 getWorldCoordinate(const float* t, const float ref_pix_size,
		const float ref_distance, const float3 &v) {
	float3 w, result;

	w = cameraBackProject(ref_pix_size, ref_distance, v);
	result = matrixVectorMultiply(t, w);

	return result;
}

__device__ __forceinline__
uint3 nearestVoxel(const VoxelVolumeCUDA &voxels, const float3 p) {

	float3 voxel_p = p - voxels.min;
	voxel_p /= voxels.size;
	return make_uint3((uint)voxel_p.x, (uint)voxel_p.y, (uint)voxel_p.z);
}

__device__ __forceinline__
float voxel(const VoxelVolumeCUDA &voxels,
		const uint x, const uint y, const uint z) {
	uint index = y * voxels.side_n2 + z * voxels.side_n + x;
	if (index < voxels.array_size) {
		return voxels.data[index];
	}
	return FLT_MIN;
}

__device__ __forceinline__
unsigned char weight(const VoxelVolumeCUDA &voxels,
		const uint x, const uint y, const uint z) {
	uint index = y * voxels.side_n2 + z * voxels.side_n + x;
	if (index < voxels.array_size) {
		return voxels.w_data[index];
	}
	return FLT_MIN;
}
