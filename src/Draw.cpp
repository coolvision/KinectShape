/*
 * Draw.cpp
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "ShapeApp.h"
#include <iostream>

void ShapeApp::rangeToWorld(Frame *f) {

	copyToDevice(camera_opt.t, f->t.getPtr(), sizeof(float) * 16);
	copyToDevice(f->dev.depth, f->host.depth, f->depth_bn);

	rangeToWorldCUDA(camera_opt, f->dev);
	sync();

	copyFromDevice(f->host.points, f->dev.points, f->points_bn);
	copyFromDevice(f->host.normals, f->dev.normals, f->points_bn);
}

// used for current surface estimate at tracking
// and for surface preview
void ShapeApp::rayMarch(Frame *f, ofMatrix4x4 *t) {

	copyToDevice(camera_opt.t, t->getPtr(), sizeof(float) * 16);

	rayMarchCUDA(camera_opt, ui_march_step, ui_march_iterations_n,
			ui_rays_step, dev_voxels, f->dev);
	sync();

	copyFromDevice(f->host.normals, f->dev.normals, f->points_bn);
	copyFromDevice(f->host.points, f->dev.points, f->points_bn);
	copyFromDevice(f->host.depth, f->dev.depth, f->dev.depth_bn);
}

void ShapeApp::resetVoxels() {

	for (int i = 0; i < voxels.array_size; i++) {
		voxels.data[i] = SDF_NONE;
		voxels.w_data[i] = 0;
	}

    copyToDevice(dev_voxels.data, voxels.data, dev_voxels.bytes_n);
	copyToDevice(dev_voxels.w_data, voxels.w_data, dev_voxels.w_bytes_n);

    sync();
}

void ShapeApp::resetVoxelWeights() {

	for (int i = 0; i < voxels.array_size; i++) {
		voxels.w_data[i] = 0;
	}

	copyToDevice(dev_voxels.w_data, voxels.w_data, dev_voxels.w_bytes_n);

    sync();
}

#define SIDEBAR_WIDTH 250

void ShapeApp::draw() {

	// shape update
//=============================================================================
	static ofNode t_inc;
	static ofMatrix4x4 t_estimate;
	static ofMatrix4x4 t_estimate_inv;
	ofMatrix4x4 t;
	ofMatrix4x4 it;

	if (ui_reset_voxels) {
		ui_reset_voxels = false;
		resetVoxels();
	}
	if (ui_reset_voxel_weights) {
		ui_reset_voxel_weights = false;
		resetVoxelWeights();
	}

	corresp.d_t = ui_distanse_threshold;
	corresp.n_t = ui_normals_threshold;

	// viewport and projection for 3d visualization
	ofViewport(SIDEBAR_WIDTH, 0, ofGetWindowWidth() - SIDEBAR_WIDTH,
			ofGetWindowHeight());
	grab_cam->begin();

	// representation of reconstruction stuff in OpenGL coords
	ofDrawAxis(500);
	drawVolume();
	drawCameraPose(&kinect, ofColor::white, t_estimate);
	drawCameraPose(&kinect, ofColor::blue,
			grab_cam->getLocalTransformMatrix());
	drawCameraPose(&kinect, ofColor::gray, curr_f.t);
	drawCameraPose(&kinect, ofColor::green, camera.getLocalTransformMatrix());

	if (kinect_on) {

		// set stride for depth data
		new_f.host.step = (int) ui_pixels_step;
		curr_f.host.step = (int) ui_pixels_step;
		est_f.host.step = (int) ui_pixels_step;

		// only update the data if there is a new frame
		kinect.update();

		// there is a new frame and we are connected
		if ((kinect.isFrameNew() && ui_update_frame) || ui_save_snapshot
				|| ui_update_continuous) {

			ui_update_frame = false;

			// reset initial tracking state
			if (ui_save_snapshot) {
				ui_save_snapshot = false;

				resetVoxels();

				camera.setPosition(0.0f, 0.0f, 0.0f);
				camera.setOrientation(ofPoint(0.0f, 0.0f, 0.0f));

				t_estimate = camera.getLocalTransformMatrix();
				t_estimate_inv = t_estimate.getInverse();

				new_f.t = camera.getLocalTransformMatrix();
				new_f.it = new_f.t.getInverse();

				est_f.t = camera.getLocalTransformMatrix();
				est_f.it = est_f.t.getInverse();

				est_f.setFromPixels(kinect.getDistancePixels());
				rangeToWorld(&est_f);
			}

			new_f.t = t_estimate;
			new_f.it = new_f.t.getInverse();

		    TIME_SAMPLE_START("rangeToWorld_new_f");
			new_f.setFromPixels(kinect.getDistancePixels());
			rangeToWorld(&new_f);
		    TIME_SAMPLE_STOP("rangeToWorld_new_f");


		    // run several correspondence iterations
		    TIME_SAMPLE_START("correspondence iterations");
		    for (int i = 0; i < ui_icp_iterations; i++) {

				float r = correspondenceIteration(&t_inc);

				if (r > 0.0f) {
					t_estimate *= t_inc.getLocalTransformMatrix();
					t_estimate_inv = t_estimate.getInverse();
				}

				new_f.t = t_estimate;
				new_f.it = new_f.t.getInverse();
				rangeToWorld(&new_f);
		    }
		    TIME_SAMPLE_STOP("correspondence iterations");

			est_f.t = t_estimate;
			est_f.it = est_f.t.getInverse();


			if (ui_update_voxels) {
			    TIME_SAMPLE_START("updateVoxelsCUDA");
				copyToDevice(camera_opt.it, est_f.it.getPtr(),
						sizeof(float) * 16);
				updateVoxelsCUDA(camera_opt, dev_voxels, new_f.dev);
				sync();
				TIME_SAMPLE_STOP("updateVoxelsCUDA");
			}

			// use raymarching to get estimated depth and normals map
			if (ui_estimate_maps) {
			    TIME_SAMPLE_START("ui_estimate_maps_rayMarch");
				rayMarch(&est_f, &est_f.t);
				TIME_SAMPLE_STOP("ui_estimate_maps_rayMarch");
			}
		}

		if (ui_show_mesh) {

			est_f.meshFromPoints(ui_show_depth);
			est_f.mesh.drawFaces();

			if (ui_icp_gpu_draw) {
				cout << "correspondence_dev" << endl;
				drawCorrespondence(&est_f, correspondence_dev);
			}
		}
	}

	grab_cam->end();

	ofViewport(0, 0, ofGetWindowWidth(), ofGetWindowHeight());

	// view the surface estimated from the voxels,
	// from an arbitrary point of view, controlled from the UI
    TIME_SAMPLE_START("drawMap");
	if (ui_show_surface_preview) {
		ofMatrix4x4 t = grab_cam->getLocalTransformMatrix();
		rayMarch(&view_f, &t);
		drawMap(&view_image, &view_f,
				ofPoint(SIDEBAR_WIDTH, 0), ui_image_scale, true);
	}
	// show  current measured data
	if (ui_show_measured_image) {
		drawMap(&image, &new_f,
				ofPoint(SIDEBAR_WIDTH,
						DEPTH_Y_RES * ui_image_scale), ui_image_scale, false);
	}
	// show the surface estimate from the point of view,
	// of the current camera pose estimate
	if (ui_show_estimated_image) {
		drawMap(&est_image, &est_f,
				ofPoint(SIDEBAR_WIDTH + DEPTH_X_RES * ui_image_scale,
						DEPTH_Y_RES * ui_image_scale), ui_image_scale, false);
	}
	TIME_SAMPLE_STOP("drawMap");

	ofSetColor(ofColor::gray);

	ofRect(0, 0, SIDEBAR_WIDTH, ofGetWindowHeight());
	gui.draw();
	TIME_SAMPLE_DRAW(SIDEBAR_WIDTH, 50);

	usleep(3000);
}

void ShapeApp::drawMap(ofImage *image, Frame *f,
		ofPoint offset, float scale, bool flip) {

	unsigned char *pixels = image->getPixels();

	if (ui_show_depth) {
		float *d = f->host.depth;
		float depth = 0;
		unsigned char r, g, b;
		for (int i = 0; i < 640 * 480; i++) {
			depth = (*d);

			if (depth == 0.0f) {
				r = 0;
				g = 0;
				b = 0;
			} else if (depth < 512.0f) {
				r = 0;
				g = depth / 2;
				b = 0;
			} else if (depth < 1024.0f) {
				r = (depth - 512.0f) / 2.0f;
				g = 0;
				b = 0;
			} else {
				r = 0;
				g = 0;
				b = (depth - 1024.0f) / 2.0f;
			}

			(*pixels++) = r;
			(*pixels++) = g;
			(*pixels++) = b;

			d++;
		}
	} else {
		float *n = f->host.normals;
		for (int i = 0; i < 640 * 480; i++) {
			(*pixels++) = (unsigned char) abs((*n++) * 255.0);
			(*pixels++) = (unsigned char) abs((*n++) * 255.0);
			(*pixels++) = (unsigned char) abs((*n++) * 255.0);
		}
	}

	image->update();
	if (flip) {
		image->mirror(false, true);
	}

	ofSetColor(ofColor::white);
	image->draw(offset.x, offset.y, image->width * scale,
			image->height * scale);
}

