/*
 * UpdateVoxels.cu
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "Common.cuh"

__global__ void updateVoxelsKernel(const CameraOptions camera_opt,
		const VoxelVolumeCUDA voxels, const FrameDataCUDA data) {

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	// find global coord of the voxel first voxel
	// on the meory-aligned line
	float3 voxel_coord;
	voxel_coord.x = voxels.min.x + x * voxels.size.x;
	voxel_coord.y = voxels.min.y + y * voxels.size.y;
	voxel_coord.z = voxels.min.z;

	int y_step = voxels.side_n2;
	int z_step = voxels.side_n;
	float *vp = &voxels.data[y * y_step + x];
	unsigned char *wp = &voxels.w_data[y * y_step + x];

	float it[16];

	for (int i = 0; i < 16; i++) {
		it[i] = camera_opt.it[i];
	}

	float factor;
	float3 v;
	int px, py;
	float depth_diff;

	// inner loop is for memory aligned line of voxels
	for (int z = 0; z < y_step; z += z_step) {

		// transform to the camera's coords
		v = matrixVectorMultiply(it, voxel_coord);
		// project it to the camera's image plane
		factor = (2 * camera_opt.ref_pix_size * v.z) /
				(camera_opt.ref_distance);
		px = v.x / factor + DEPTH_X_RES / 2;
		py = v.y / factor + DEPTH_Y_RES / 2;

		if (px > 0 && px < data.width &&
			py > 0 && py < data.height) {

			// is negative, to align ki-nect coordinates with OpenGL
			float depth = -data.depth[py * data.width + px] / 1000.0f;

			if (depth < camera_opt.max[2] && depth > camera_opt.min[2]) {

				// get the ray length from the depth value
				// first, get the vector in camera coordinate system
				// which is directed along the ray projected from a pixel
				// and has unit z distance from the image plane
				float factor = 2 * camera_opt.ref_pix_size /
								   camera_opt.ref_distance;
				float3 unit_z;
				unit_z.x = (px - DEPTH_X_RES / 2) * factor;
				unit_z.y = (py - DEPTH_Y_RES / 2) * factor;
				unit_z.z = 1.0;

				// get the length of this vector
				float unit_z_ray_length = length(unit_z);

				// can use this length to transform depth value into
				// the distance along the ray
				float measured_ray_length = depth * unit_z_ray_length;

				float depth_diff = (-length(v)) - measured_ray_length;

				// now if difference is positive, this is a free space
				// store the truncated distance
				float tsdf = FLT_MIN;
				if (depth_diff > 0.0f) {
					if (depth_diff < SDF_T) {
						tsdf = depth_diff;
					} else {
						tsdf = SDF_T;
					}
				} else {
					// negative distance, update if it is within
					// truncation distance, otherwise do not measure at all
					if (depth_diff > -SDF_T) {
						tsdf = depth_diff;
					}
				}
				// make a weighted averaging of the distance
				if (tsdf != FLT_MIN) {
					if (*vp == -FLT_MAX) {
						// first measurement, just use the distance
						if ((*wp) > 10) // magic number, 2FIX
							*vp = tsdf;
					} else {
						// update weighted average
						if ((*wp) > 10) // magic number, 2FIX
							*vp = ((*wp) * (*vp) + tsdf) / ((*wp) + 1);
						//*vp = tsdf;
					}

					// also update the weight
					if ((*wp) < 255) (*wp)++;
				}
			}
		}

		if ((*wp) < 255) (*wp)++;

		vp += z_step; // step through the memory
		wp += z_step; // weights
		voxel_coord.z += voxels.size.z; // and the mm space
	}
}

void updateVoxelsCUDA(CameraOptions camera_opt,
		VoxelVolumeCUDA voxels, FrameDataCUDA data) {

	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks(divUp(voxels.side_n, threadsPerBlock.x),
				divUp(voxels.side_n, threadsPerBlock.y));

	updateVoxelsKernel<<<numBlocks, threadsPerBlock>>>(camera_opt,
			voxels, data);
}
