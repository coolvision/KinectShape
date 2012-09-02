/*
 * RayMarch.cu
 *
 *  Created on: 2012
 *      Author: s
 */

#include "Common.cuh"

__global__ void rayMarchKernel(const CameraOptions camera_opt,
		const float march_step, const float march_iterations_n, const int step,
		const VoxelVolumeCUDA voxels, const FrameDataCUDA data) {

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	uint i = y * data.width + x;
	uint p_i = i * 3;

	// ray direction in image coordinates
	float3 init_start = make_float3(x, y, 0);
	float3 direction = make_float3(0.0, 0.0, -1.0);
	float3 direction_p = init_start + direction;

	// transform to the world
	float3 start;

	start = getWorldCoordinate(camera_opt.t,
			camera_opt.ref_pix_size, camera_opt.ref_distance, init_start);
	direction_p = getWorldCoordinate(camera_opt.t,
			camera_opt.ref_pix_size, camera_opt.ref_distance,
			direction_p);

	direction = direction_p - start;
	direction = normalize(direction); // again unit vector

	// now, along this ray's line step for some iterations,
	// with a fixed step, say, 10cm
	int stop_status = ITERATIONS_EXPIRED;
	int m = 0;
	float previous_d = FLT_MAX;
	float current_d = FLT_MAX;
	bool is_inside = false;
	float3 p;
	float3 normal;
	uint3 v;
	float approx_d;

	for (m = 0; m < march_iterations_n; m++) {

		// for each step, there is a marched point
		p = start + direction * m * march_step;

		// get the nearest voxel coordinate
		// 2DO use trilinear interpolation
		v = nearestVoxel(voxels, p);

		if (v.x > 0 && v.y > 0 && v.z > 0 && v.x < voxels.side_n
				&& v.y < voxels.side_n && v.z < voxels.side_n) {
			is_inside = true;
		} else {
			if (is_inside) {
				// if the point was inside of the volume, and then
				// out, then can stop the walk as well
				stop_status = MISS_VOLUME;
				break;
			}
		}

		if (is_inside) {
			// if it is inside, check the nearest voxel value
			current_d = voxel(voxels, v.x, v.y, v.z);

			if (current_d != -FLT_MAX &&
				current_d <= 0.0f && previous_d > 0.0f) {

				approx_d = m * march_step + current_d;

				// get the approximated voxel coordinate
				p = start + direction * approx_d;
				v = nearestVoxel(voxels, p);

				// now estimate the normal direction
				// should make this piece more compact
				normal.x = voxel(voxels, v.x + 1, v.y, v.z)
						- voxel(voxels, v.x - 1, v.y, v.z);
				normal.y = voxel(voxels, v.x, v.y + 1, v.z)
						- voxel(voxels, v.x, v.y - 1, v.z);
				normal.z = voxel(voxels, v.x, v.y, v.z + 1)
						- voxel(voxels, v.x, v.y, v.z - 1);

				normal = normalize(normal);

				stop_status = STOP_AT_SURFACE;
				break;
			}

			if (previous_d != -FLT_MAX &&
					current_d >= 0.0f && previous_d < 0.0f) {
				stop_status = NEGATIVE_CROSS;
				break;
			}

			previous_d = current_d;
		}
	}

	// draw marching result
	if (stop_status == STOP_AT_SURFACE) {

		data.normals[p_i] = normal.x;
		data.normals[p_i+1] = normal.y;
		data.normals[p_i+2] = normal.z;

		data.points[p_i] = p.x;
		data.points[p_i+1] = p.y;
		data.points[p_i+2] = p.z;

		// output is the same as kinect output: positive millimeters
		data.depth[i] = approx_d * 1000.0f;

	} else {

		data.normals[p_i] = 0.0f;
		data.normals[p_i+1] = 0.0f;
		data.normals[p_i+2] = 0.0f;

		data.points[p_i] = 0.0f;
		data.points[p_i+1] = 0.0f;
		data.points[p_i+2] = 0.0f;

		data.depth[i] = 0.0f;
	}
}

void rayMarchCUDA(CameraOptions camera_opt,
		float march_step, float march_iterations_n, int step,
		VoxelVolumeCUDA voxels, FrameDataCUDA data) {

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(divUp(data.width, threadsPerBlock.x),
				divUp(data.height, threadsPerBlock.y));

	rayMarchKernel<<<numBlocks, threadsPerBlock>>>(camera_opt,
			march_step, march_iterations_n, step,
			voxels, data);
}

