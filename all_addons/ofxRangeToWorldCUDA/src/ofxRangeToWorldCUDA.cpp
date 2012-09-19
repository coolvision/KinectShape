/*
 * ofxRangeToWorldCUDA.cpp
 *
 *  Created on: 2012
 *      Author: sk
 */

#pragma once

#include <cutil_inline.h>

#include "ofxRangeToWorldCUDA.h"

void sync() {
	cudaDeviceSynchronize();
}

void copyToDevice(void *to, void *from, size_t size, bool do_sync) {
	cutilSafeCall(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
	if (do_sync) sync();
}
void copyFromDevice(void *to, void *from, size_t size, bool do_sync) {
	cutilSafeCall(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
	if (do_sync) sync();
}

void rangeToWorldCUDA(CameraOptions camera_opt, FrameDataCUDA data);

void rangeToWorld(CameraOptions *camera_opt, Frame *f) {

	copyToDevice(camera_opt->t, f->t.getPtr(), sizeof(float) * 16);
	copyToDevice(f->dev.depth, f->host.depth, f->depth_bn);

	rangeToWorldCUDA(*camera_opt, f->dev);
	sync();

	copyFromDevice(f->host.points, f->dev.points, f->points_bn);
	copyFromDevice(f->host.normals, f->dev.normals, f->points_bn);
}
