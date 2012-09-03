/*
 * Setup.cpp
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "ShapeApp.h"

#include <cutil_inline.h> // includes cuda.h and cuda_runtime_api.h

int divUp(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void ShapeApp::setupUI() {

	grab_cam = new ofxGrabCam(false);
	grab_cam->reset();
	grab_cam->setPosition(1.0, 1.0, 1.0);
	grab_cam->lookAt(ofVec3f(0.0f, 0.0f, 0.0f));

	ui_save_snapshot = false;

	gui.setDefaultKeys(true);
	gui.setDraw(true);

	gui.addFPSCounter();

	gui.addButton("update_frame", ui_update_frame);
	gui.addToggle("update_continuous", ui_update_continuous);
	gui.addSlider("image_scale", ui_image_scale, 0.2, 1.0);
	gui.addToggle("mesh", ui_show_mesh);
	gui.addSlider("pixels_step", ui_pixels_step, 1, 10);
	gui.addToggle("surface_preview", ui_show_surface_preview);
	gui.addToggle("measured_image", ui_show_measured_image);
	gui.addToggle("estimated_image", ui_show_estimated_image);
	gui.addToggle("depth", ui_show_depth);

	gui.addTitle("voxels");
	gui.addToggle("update_voxels", ui_update_voxels);
	gui.addToggle("estimate_maps", ui_estimate_maps);
	gui.addButton("reset", ui_reset_voxels);
	gui.addButton("reset_weights", ui_reset_voxel_weights);

	gui.addTitle("rays");
	gui.addSlider("march_step", ui_march_step, 0.0005, 0.1);
	gui.addSlider("iterations_n", ui_march_iterations_n, 10, 1000);

	gui.addTitle("ICP");
	gui.addButton("save_snapshot", ui_save_snapshot);
	gui.addSlider("icp_iterations", ui_icp_iterations, 0, 10);
	gui.addSlider("normals_threshold", ui_normals_threshold, 0.0f, 2 * M_PI);
	gui.addSlider("distanse_threshold", ui_distanse_threshold, 0.0f, 0.5f);
	gui.addToggle("draw", ui_icp_gpu_draw);

	gui.loadFromXML();
	gui.show();
}

void ShapeApp::setup() {

	// setup kinect
//=============================================================================
	kinect.setRegistration(true);
	kinect.init();
	if (kinect.open()) {
		kinect_on = true;
	} else {
		kinect_on = false;
	}

	while (!kinect.isConnected());

	ofSetFrameRate(30);
	TIME_SAMPLE_SET_FRAMERATE(30.0f);

	// setup UI
//=============================================================================
	setupUI();

	// setup tracking data objects
//=============================================================================
	// CPU
	curr_f.kinect = &kinect;
	curr_f.init(DEPTH_X_RES, DEPTH_Y_RES);
	curr_f.allocateHost();

	new_f.kinect = &kinect;
	new_f.init(DEPTH_X_RES, DEPTH_Y_RES);
	new_f.allocateHost();

	est_f.kinect = &kinect;
	est_f.init(DEPTH_X_RES, DEPTH_Y_RES);
	est_f.allocateHost();

	view_f.init(DEPTH_X_RES, DEPTH_Y_RES);
	view_f.allocateHost();

	image.allocate(DEPTH_X_RES, DEPTH_Y_RES, OF_IMAGE_COLOR);
	view_image.allocate(DEPTH_X_RES, DEPTH_Y_RES, OF_IMAGE_COLOR);
	est_image.allocate(DEPTH_X_RES, DEPTH_Y_RES, OF_IMAGE_COLOR);

	// GPU
	curr_f.allocateDevice();
	new_f.allocateDevice();
	est_f.allocateDevice();
	view_f.allocateDevice();

	// data for ICP
//=============================================================================
	// GPU
	corresp.blocks_x =
			divUp(DEPTH_X_RES, CORRESPONDENCE_BLOCK_X);
	corresp.blocks_y =
			divUp(DEPTH_Y_RES, CORRESPONDENCE_BLOCK_Y);
	corresp.blocks_n = corresp.blocks_x * corresp.blocks_y;
	corresp.AtA_dev_size = AtA_SIZE * corresp.blocks_n;
	corresp.Atb_dev_size = Atb_SIZE * corresp.blocks_n;

	cudaMalloc((void **) &corresp.AtA_dev,
			corresp.AtA_dev_size * sizeof(float));
	cudaMalloc((void **) &corresp.Atb_dev,
			corresp.Atb_dev_size * sizeof(float));

	cudaMalloc((void **) &corresp.points_dev, curr_f.host.points_bn);

	// CPU
	corresp.AtA_host = (float *)malloc(corresp.AtA_dev_size * sizeof(float));
	corresp.Atb_host = (float *)malloc(corresp.Atb_dev_size * sizeof(float));

	corresp.AtA_sum = (float *)malloc(AtA_SIZE * sizeof(float));
	corresp.Atb_sum = (float *)malloc(Atb_SIZE * sizeof(float));

	correspondence_host = (float *) malloc(curr_f.host.points_bn);
	correspondence_dev = (float *) malloc(curr_f.host.points_bn);
	corresp.points_host = (float *)malloc(curr_f.host.points_bn);

	// voxel data
//=============================================================================
	// CPU
	min.set(-0.5, -0.5, -1.5);
	max.set(0.5, 0.5, -0.5);

	voxels.min = min;
	voxels.side_n = 256;
	voxels.size = (max - min) / (float)voxels.side_n;

	voxels.array_size = voxels.side_n * voxels.side_n * voxels.side_n;
	voxels.bytes_n = sizeof(float) * voxels.array_size;
	voxels.data = (float *)malloc(voxels.bytes_n);

	voxels.w_bytes_n = sizeof(unsigned char) * voxels.array_size;
	voxels.w_data = (unsigned char *)malloc(voxels.w_bytes_n);

	// GPU
	cudaMalloc((void **) &camera_opt.t, sizeof(float) * 16);
	cudaMalloc((void **) &camera_opt.it, sizeof(float) * 16);
	camera_opt.ref_pix_size = kinect.getRefPixelSize();
	camera_opt.ref_distance = kinect.getRefDistance();
	setFloat3(camera_opt.min, min);
	setFloat3(camera_opt.max, max);

	cudaMalloc((void **) &dev_voxels.data, voxels.bytes_n);
	cudaMalloc((void **) &dev_voxels.w_data, voxels.w_bytes_n);
	setFloat3(&dev_voxels.min, voxels.min);
	setFloat3(&dev_voxels.size, voxels.size);

	dev_voxels.side_n = voxels.side_n;
	dev_voxels.side_n2 = dev_voxels.side_n * dev_voxels.side_n;
	dev_voxels.array_size = voxels.array_size;
	dev_voxels.bytes_n = voxels.bytes_n;
	dev_voxels.w_bytes_n = voxels.w_bytes_n;

	resetVoxels();
}

void setFloat3(float3 *f, ofPoint p) {
	f->x = p.x;
	f->y = p.y;
	f->z = p.z;
}

void setFloat3(float *f, ofPoint p) {
	f[0] = p.x;
	f[1] = p.y;
	f[2] = p.z;
}

void ShapeApp::exit() {

	delete grab_cam;

	free(correspondence_host);

	cudaFree(camera_opt.t);
	cudaFree(camera_opt.it);
	cudaFree(dev_voxels.data);
	cudaFree(dev_voxels.w_data);
	cudaFree(corresp.AtA_dev);
	cudaFree(corresp.Atb_dev);
	cudaFree(corresp.points_dev);

	free(voxels.data);
	free(voxels.w_data);

	curr_f.releaseHost();
	new_f.releaseHost();
	est_f.releaseHost();
	view_f.releaseHost();

	curr_f.releaseDevice();
	new_f.releaseDevice();
	est_f.releaseDevice();
	view_f.releaseDevice();

	kinect.close();

	cout << "ShapeApp::exit()" << endl;
}


