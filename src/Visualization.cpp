/*
 * Visualization.cpp
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "ShapeApp.h"

void drawCameraPose(ofxKinect *kinect, ofColor color,
        ofMatrix4x4 transform_matrix) {

	ofPoint near[4];
	ofPoint far[4];
	ofPoint camera_near[4];
	ofPoint camera_far[4];
	ofPoint world_near[4];
	ofPoint world_far[4];

	//int width = kinect->getDepthPixelsRef().getWidth();
	//int height = kinect->getDepthPixelsRef().getHeight();
    int width;
    int height;

	// so, there are some points for display of the camera pose
	near[0].set(0, 0, 0.0f);
	near[1].set(0, height, 0.0f);
	near[2].set(width, height, 0.0f);
	near[3].set(width, 0, 0.0f);
	far[0].set(0, 0, -1500);
	far[1].set(0, height, -1500);
	far[2].set(width, height, -1500);
	far[3].set(width, 0, -1500);

	// first, transform some points into camera coordinates
	for (int i = 0; i < 4; i++) {
        camera_near[i] = kinect->getWorldCoordinateAt(near[i].x, near[i].y,
                near[i].z);
        camera_far[i] = kinect->getWorldCoordinateAt(far[i].x, far[i].y,
                far[i].z);

		camera_near[i] /= 1000.0;
		camera_far[i] /= 1000.0;
	}

	ofSetLineWidth(1.0);

	// now transform this points
	for (int i = 0; i < 4; i++) {
		world_near[i] = camera_near[i] * transform_matrix;
		world_far[i] = camera_far[i] * transform_matrix;
	}
	ofSetColor(color);
	for (int i = 0; i < 4; i++) {
		ofLine(world_near[i], world_far[i]);
	}
	ofLine(world_far[0], world_far[1]);
	ofLine(world_far[1], world_far[2]);
	ofLine(world_far[2], world_far[3]);
	ofLine(world_far[3], world_far[0]);
}

void ShapeApp::drawVolume() {

	float width = max.x - min.x;
	float height = max.y - min.y;

	ofPoint near_v[4];
	ofPoint far_v[4];
	near_v[0] = min;
	near_v[1].set(min.x + width, min.y, min.z);
	near_v[2].set(min.x + width, min.y + height, min.z);
	near_v[3].set(min.x, min.y + height, min.z);

	far_v[0].set(max.x - width, max.y - height, max.z);
	far_v[1].set(max.x, max.y - height, max.z);
	far_v[2] = max;
	far_v[3].set(max.x - width, max.y, max.z);

	ofSetColor(ofColor::green);
	ofSetLineWidth(1.0);
	ofLine(near_v[0], near_v[1]);
	ofLine(near_v[1], near_v[2]);
	ofLine(near_v[2], near_v[3]);
	ofLine(near_v[3], near_v[0]);

	ofLine(far_v[0], far_v[1]);
	ofLine(far_v[1], far_v[2]);
	ofLine(far_v[2], far_v[3]);
	ofLine(far_v[3], far_v[0]);

	for (int i = 0; i < 4; i++) {
		ofLine(near_v[i], far_v[i]);
	}
}
