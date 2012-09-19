#pragma once

#include "ofMain.h"
#include "ofxKinect.h"
#include "ofxGrabCam.h"
#include "ofxSimpleGuiToo.h"
#include "ofxTimeMeasurements.h"

#include "Utils.h"
#include "Frame.h"
#include "Voxels.h"
#include "Settings.h"

#include "VoxelsCUDA.h"
#include "ofxRangeToWorldCUDA.h"

#include "libfreenect-registration.h"

#include <iostream>
#undef Success // fix for Eigen and X11 defines conflict
#include <Eigen/Dense>

#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h

// simple utils for CUDA code
void setFloat3(float3 *f, ofPoint p);
void setFloat3(float *f, ofPoint p);


void drawCameraPose(ofxKinect *kinect,
		ofColor color, ofMatrix4x4 transform_matrix);

void rayMarchCUDA(CameraOptions camera_opt, float march_step,
		float march_iterations_n, int step, VoxelVolumeCUDA voxels,
		FrameDataCUDA data);
void updateVoxelsCUDA(CameraOptions camera_opt, VoxelVolumeCUDA voxels,
		FrameDataCUDA data);
void findCorrespondenceCUDA(CameraOptions camera_opt,
		FrameDataCUDA curr_data, FrameDataCUDA new_data,
		CorrespondenceCUDA corresp);

class ShapeApp : public ofBaseApp {
//class ShapeApp : public ofxFensterListener {
public:
	// status
//=============================================================================
	bool kinect_on;

	// tracking and reconstruction data
//=============================================================================

	Frame curr_f; // in first prototype, just a snapshot
	Frame new_f; // new data: estimate transform, add up to the volume
	Frame est_f; // estimated from voxels for ICP tracking
	Frame view_f; // estimated from voxels for surface viewing

	CameraOptions camera_opt;

	// voxels on CPU
	VoxelVolume voxels;
	// voxels on GPU
	VoxelVolumeCUDA dev_voxels;

	// current measured data
	ofImage image;

	// surface estimate from arbitrary point of view
	ofImage view_image;

	// surface estimate from the current
	// camera pose estimate point of view
	ofImage est_image;


//=============================================================================

	void resetVoxels();
	void resetVoxelWeights();
	void rayMarch(Frame *f, ofMatrix4x4 *t);

	void drawMap(ofImage *image, Frame *f,
			ofPoint offset, float scale, bool flip);

	// correspondence on GPU
//=============================================================================
	CorrespondenceCUDA corresp;

	// correspondence and ICP on CPU
//=============================================================================
	float *correspondence_host;
	float *correspondence_dev;
	float AtA[6 * 6];
	float Atb[6];

	float correspondenceIteration(ofNode *t);

	float findCorrespondence(FrameData *current_data, FrameData *new_data,
			float *t_new_inv,
			float *correspondence,
			float d_t, float n_t,
			float *AtA, float *Atb);

	// visualization
//=============================================================================
	void drawVoxels();
	void drawVolume();
	void drawCorrespondence(Frame *frame, float *correspondence);


	// data acquisition and visualization
//=============================================================================
	unsigned long rangeToWorld_time;
	unsigned long updateVoxelsCUDA_time;
	unsigned long CUDA_raymarch_time;
	unsigned long correspondenceIteration_time;

	ofxKinect kinect;
	float angle;

	bool ui_update_frame;
	bool ui_update_continuous;

	bool ui_show_mesh;
	bool ui_show_surface_preview;
	bool ui_show_measured_image;
	bool ui_show_estimated_image;
	bool ui_show_depth;

	// test and utility camera
	ofCamera camera;
	bool ui_reset;

	// rays
	int ui_rays_step;
	float ui_image_scale;
	float ui_march_step;
	int ui_march_iterations_n;

	// ICP
	int ui_icp_iterations;
	bool ui_save_snapshot;
	int ui_pixels_step;
	float ui_normals_threshold;
	float ui_distanse_threshold;
	bool ui_icp_gpu_draw;

	// voxels
	bool ui_update_voxels;
	bool ui_estimate_maps;
	bool ui_reset_voxels;
	bool ui_reset_voxel_weights;

	// weighted averaging o the distances
	float ui_voxels_alpha;
	// distance function truncation
	float ui_voxels_sdf_t;


	// thresholds for the depth data
	// use only the points inside of the 3d box
	ofPoint min;
	ofPoint max;

	ofxGrabCam *grab_cam;

	void setupUI();

//============================================================================
	void setup();
	void update();
	void draw();

	void exit();

	void keyPressed(ofKeyEventArgs &args);
	void keyPressed(int key);
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void windowResized(int w, int h);

};
