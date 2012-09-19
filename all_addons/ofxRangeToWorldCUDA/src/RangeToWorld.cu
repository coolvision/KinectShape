/*
 * RangeToWorld.cu
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "Common.cuh"

__global__ void rangeToWorldKernel(const CameraOptions camera_opt,
		const FrameDataCUDA data) {

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	uint i = y * data.width + x;
	uint p_i = i * 3;

	// change depth direction, so it would correspond to openGL coords
	float3 d = make_float3(x, y, -data.depth[i]);

	float3 p = getWorldCoordinate(camera_opt.t,
			camera_opt.ref_pix_size, camera_opt.ref_distance, d);

	// 3d space filtering
	if (p.x > camera_opt.min[0] && p.x < camera_opt.max[0] &&
		p.y > camera_opt.min[1] && p.y < camera_opt.max[1] &&
		p.z > camera_opt.min[2] && p.z < camera_opt.max[2]) {

		data.points[p_i] = p.x;
		data.points[p_i + 1] = p.y;
		data.points[p_i + 2] = p.z;

	} else {

		data.points[p_i] = 0.0f;
		data.points[p_i + 1] = 0.0f;
		data.points[p_i + 2] = 0.0f;
	}
}

__global__ void getNormalsKernel(const CameraOptions camera_opt,
		const FrameDataCUDA data) {

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	uint i = y * data.width + x;
	uint p_i = i * 3;

	// estimate a normal
	float3 p1 = getPoint(data.points, x + 1, y, data.width);
	float3 p2 = getPoint(data.points, x, y + 1, data.width);
	float3 p3 = getPoint(data.points, x, y, data.width);

	float3 normal;

	if (p1.z == 0.0 || p2.z == 0 || p3.z == 0) {

		normal.x = 0.0f;
		normal.y = 0.0f;
		normal.z = 0.0f;

	} else {

		float3 v1 = p1 - p3;
		float3 v2 = p2 - p3;

		normal = cross(v1, v2);
		normal = normalize(normal);
	}

	data.normals[p_i] = normal.x;
	data.normals[p_i + 1] = normal.y;
	data.normals[p_i + 2] = normal.z;
}

void rangeToWorldCUDA(CameraOptions camera_opt, FrameDataCUDA data) {

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(divUp(data.width, threadsPerBlock.x),
					divUp(data.height, threadsPerBlock.y));

	rangeToWorldKernel<<<numBlocks, threadsPerBlock>>>(camera_opt, data);

	cutilSafeCall(cutilDeviceSynchronize());

	getNormalsKernel<<<numBlocks, threadsPerBlock>>>(camera_opt, data);

	cutilSafeCall(cutilDeviceSynchronize());
}

