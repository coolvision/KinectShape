/*
 * Update.cpp
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "ShapeApp.h"

void ShapeApp::update() {

}

void ShapeApp::keyPressed(ofKeyEventArgs &args) {

	switch (args.key) {

	case 's':
		ui_save_snapshot = true;
		break;

	case 'w':
		kinect.enableDepthNearValueWhite(!kinect.isDepthNearValueWhite());
		break;

	case 'o':
		kinect.setCameraTiltAngle(angle); // go back to prev tilt
		kinect.open();
		break;

	case 'c':
		kinect.setCameraTiltAngle(0); // zero the tilt
		kinect.close();
		break;

	case OF_KEY_UP:
		angle++;
		if (angle > 30)
			angle = 30;
		kinect.setCameraTiltAngle(angle);
		break;

	case OF_KEY_DOWN:
		angle--;
		if (angle < -30)
			angle = -30;
		kinect.setCameraTiltAngle(angle);
		break;
	case 270:
		angle++;
		if (angle > 30)
			angle = 30;
		kinect.setCameraTiltAngle(angle);
		break;

	case 269:
		angle--;
		if (angle < -30)
			angle = -30;
		kinect.setCameraTiltAngle(angle);
		break;
	}
}

void ShapeApp::keyPressed(int key) {
}

void ShapeApp::mouseDragged(int x, int y, int button) {
}

void ShapeApp::mousePressed(int x, int y, int button) {
}

void ShapeApp::mouseReleased(int x, int y, int button) {
}

void ShapeApp::windowResized(int w, int h) {
}
