/*
 * Frame.cpp
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "Frame.h"
#include <cuda.h>
#include "cuda_runtime_api.h"

void Frame::init(int width, int height) {

	host.depth = NULL;
	host.points = NULL;
	host.normals = NULL;
	host.width = width;
	host.height = height;
	host.step = 1;

	dev.depth = NULL;
	dev.points = NULL;
	dev.normals = NULL;
	dev.width = width;
	dev.height = height;
	dev.step = 1;

	depth_bn = width * height * sizeof(float);
	points_bn = width * height * 3 * sizeof(float);
}

void Frame::setFromPixels(float *pixels) {

	size_t depth_bn = host.width * host.height * sizeof(float);

	memcpy(host.depth, pixels, depth_bn);
}

void Frame::allocateDevice() {

	dev.depth_bn = depth_bn;
	dev.points_bn = points_bn;

	cudaMalloc((void **) &dev.depth, depth_bn);
	cudaMalloc((void **) &dev.points, points_bn);
	cudaMalloc((void **) &dev.normals, points_bn);
}

void Frame::releaseDevice() {

	cudaFree(dev.depth);
	cudaFree(dev.points);
	cudaFree(dev.normals);
}

void Frame::allocateHost() {

	host.depth_bn = depth_bn;
	host.points_bn = points_bn;

	host.depth = (float *) malloc(host.depth_bn);
	host.points = (float *) malloc(host.points_bn);
	host.normals = (float *) malloc(host.points_bn);
}

void Frame::releaseHost() {

	free(host.depth);
	free(host.points);
	free(host.normals);
}

void Frame::meshFromPoints(bool draw_normals, ofxKinect *kinect) {

	// make a mesh from returned data
	ofPoint p;
	ofPoint normal;
	ofPoint local_points[2][2];

	mesh.clear();
	mesh.setMode(OF_PRIMITIVE_TRIANGLES);

	for (int y = host.step;
			y < host.height - host.step; y += host.step) {
		for (int x = host.step;
				x < host.width - host.step; x += host.step) {

			bool zero_depth_found = false;
			for (int i = 0; i <= 1; i++) {
				for (int j = 0; j <= 1; j++) {
					host.getPoint(&local_points[i][j],
							x + i * host.step, y + j * host.step);
					if (local_points[i][j].z == 0.0) {
						zero_depth_found = true;
						break;
					}
				}
			}

			if (zero_depth_found)
				continue;


			host.getNormal(&normal, x, y);

			ofColor color;
			if (draw_normals) {
				color.set(abs(normal.x) * 255.0,
						abs(normal.y) * 255.0,
						abs(normal.z) * 255.0, 200.0f);
			} else {
				color = kinect->getColorAt(x, y);
			}

			mesh.addVertex(local_points[0][0]);
			mesh.addColor(color);
			mesh.addNormal(normal);
			mesh.addVertex(local_points[1][0]);
			mesh.addColor(color);
			mesh.addNormal(normal);
			mesh.addVertex(local_points[0][1]);
			mesh.addColor(color);
			mesh.addNormal(normal);

			mesh.addVertex(local_points[1][1]);
			mesh.addColor(color);
			mesh.addNormal(normal);
			mesh.addVertex(local_points[0][1]);
			mesh.addColor(color);
			mesh.addNormal(normal);
			mesh.addVertex(local_points[1][0]);
			mesh.addColor(color);
			mesh.addNormal(normal);
		}
	}
}

void Frame::drawMesh() {

	// draw the mesh
	mesh.drawFaces();
}

void FrameData::getPoint(ofPoint *point, int x, int y) {

	int index = (y * width + x) * 3;
	point->x = points[index];
	point->y = points[index + 1];
	point->z = points[index + 2];
}

void FrameData::getNormal(ofPoint *point, int x, int y) {

	int index = (y * width + x) * 3;
	point->x = normals[index];
	point->y = normals[index + 1];
	point->z = normals[index + 2];
}
