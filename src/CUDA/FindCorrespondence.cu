/*
 * FindCorrespondence.cu
 *
 *  Created on: 2012
 *      Author: sk
 */


#include "Common.cuh"

__global__ void findCorrespondenceKernel(const CameraOptions camera_opt,
		const FrameDataCUDA curr_data, const FrameDataCUDA new_data,
		const float d_t, const float n_t,
		float *AtA, float *Atb,
		float *points) {

	// store each calculate element for parallel addition within the block
	__shared__ float s_AtA[CORRESPONDENCE_BLOCK_SIZE];
	__shared__ float s_Atb[CORRESPONDENCE_BLOCK_SIZE];

	// get the pixel index
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * curr_data.width + x;
	uint p_i = i * 3;

	// index inside of the block
	uint block_i = blockIdx.y * gridDim.x + blockIdx.x;
	uint thread_i = blockDim.x * threadIdx.y + threadIdx.x;

	float A_row[6];
	float b_value;

	for (int i = 0; i < 6; i++) {
		A_row[i] = 0.0f;
	}
	b_value = 0.0f;

	// find correspondence for this pixel
	float3 v =	getPoint(curr_data.points, x, y, curr_data.width);
	float3 curr_n = getPoint(curr_data.normals, x, y, curr_data.width);

	if (v.z > camera_opt.min[2] &&
		v.z < camera_opt.max[2]) {

		// project this point onto the camera's image plane
		float3 camera_p = matrixVectorMultiply(camera_opt.it, v);

		float factor = (2 * camera_opt.ref_pix_size * camera_p.z) /
				(camera_opt.ref_distance);
		int px = camera_p.x / factor + DEPTH_X_RES / 2;
		int py = camera_p.y / factor + DEPTH_Y_RES / 2;

		// only for the points that are projected into the viewport
		if (px >= 0 && px < curr_data.width &&
			py >= 0 && py < curr_data.height) {

			float3 new_p = getPoint(new_data.points, px, py, curr_data.width);
			float3 new_n = getPoint(new_data.normals, px, py, curr_data.width);
			float3 c = new_p - v;

			// don't bother with points outside of the specified depth range
			if (new_p.z > camera_opt.min[2] &&
				new_p.z < camera_opt.max[2]) {

				float3 n1 = normalize(new_n);
				float3 n2 = normalize(curr_n);
				float angle = acos(dot(n1, n2));

				if (abs(angle) > n_t) {
					setPoint(make_float3(100.0f, 100.0f, angle),
							points, x, y, curr_data.width);
				} else {
					setPoint(c, points, x, y, curr_data.width);
				}

				float distance = length(c);
				if (distance < d_t && abs(angle) < n_t) {

					// calculate row of A matrix
					// 2DO cross-product, can write more compact
					A_row[0] = new_p.z * curr_n.y - new_p.y * curr_n.z;
					A_row[1] = new_p.x * curr_n.z - new_p.z * curr_n.x;
					A_row[2] = new_p.y * curr_n.x - new_p.x * curr_n.y;
					A_row[3] = curr_n.x;
					A_row[4] = curr_n.y;
					A_row[5] = curr_n.z;

					// calculate value of the b vector
					// 2DO dot product, can write more compact
					b_value = curr_n.x * (v.x - new_p.x)
							+ curr_n.y * (v.y - new_p.y)
							+ curr_n.z * (v.z - new_p.z);
				}
			}
		}
	}

	// now add up and store AtA matrix values: left part of the normal system
	// (only upper half of the symmetric matrix)
	for (int i = 0; i < 6; i++) {
		for (int j = i; j < 6; j++) {
			// add up values one-by-one, once they are computed by all threads
			s_AtA[thread_i] = A_row[i] * A_row[j];
			__syncthreads();

			// run reduction
			for (unsigned int s = 1; s < CORRESPONDENCE_BLOCK_SIZE; s *= 2) {
				if (thread_i % (2 * s) == 0) {
					s_AtA[thread_i] += s_AtA[thread_i + s];
				}
				__syncthreads();
			}

			// save result to a global location
			if (thread_i == 0) {
				AtA[block_i * AtA_SIZE + i * 6 + j] = s_AtA[0];
			}
		}
	}

	// same for the Atb vector, right part of the normal system
	for (int i = 0; i < 6; i++) {
		// add up values one-by-one, once they are computed by all threads
		s_Atb[thread_i] = b_value * A_row[i];
		__syncthreads();

		// run reduction
		for (unsigned int s = 1; s < CORRESPONDENCE_BLOCK_SIZE; s *= 2) {
			if (thread_i % (2 * s) == 0) {
				s_Atb[thread_i] += s_Atb[thread_i + s];
			}
			__syncthreads();
		}

		// save result to a global location
		if (thread_i == 0) {
			Atb[block_i * Atb_SIZE + i] = s_Atb[0];
		}
	}

}

void findCorrespondenceCUDA(CameraOptions camera_opt,
		FrameDataCUDA curr_data, FrameDataCUDA new_data,
		CorrespondenceCUDA corresp) {

	dim3 threadsPerBlock(CORRESPONDENCE_BLOCK_X, CORRESPONDENCE_BLOCK_Y);
	dim3 numBlocks(corresp.blocks_x, corresp.blocks_y);

	findCorrespondenceKernel<<<numBlocks, threadsPerBlock>>>(camera_opt,
			curr_data, new_data, corresp.d_t, corresp.n_t,
			corresp.AtA_dev, corresp.Atb_dev,
			corresp.points_dev);
}
