/*
 * ICP.cpp
 *
 *  Created on: 2012
 *      Author: sk
 */

#include "ShapeApp.h"

float ShapeApp::correspondenceIteration(ofNode *t) {

	float correspondence_ratio;

	if (ui_icp_gpu_draw) {
		for (int y = 0; y < est_f.host.height; y += est_f.host.step) {
			for (int x = 0; x < est_f.host.width; x += est_f.host.step) {
				setPoint(correspondence_dev, x, y,
						est_f.host.width, 0.0, 0.0, 0.0);
			}
		}
		copyToDevice(corresp.points_dev, correspondence_dev,
				new_f.points_bn);
	}

	TIME_SAMPLE_START("correspondence_GPU");

	// points/normals data should be on the device already
	copyToDevice(camera_opt.it, est_f.it.getPtr(), sizeof(float) * 16);

	findCorrespondenceCUDA(camera_opt, est_f.dev, new_f.dev, corresp);
	sync();

	copyFromDevice(corresp.AtA_host, corresp.AtA_dev,
			   corresp.AtA_dev_size * sizeof(float));
	copyFromDevice(corresp.Atb_host, corresp.Atb_dev,
			   corresp.Atb_dev_size * sizeof(float));

	// should measure it
	correspondence_ratio = 1.0;


	for (int i = 0; i < AtA_SIZE; i++) {
		corresp.AtA_sum[i] = 0.0f;
	}
	for (int i = 0; i < Atb_SIZE; i++) {
		corresp.Atb_sum[i] = 0.0f;
	}
	for (int i = 0; i < corresp.blocks_n; i++) {
		for (int j = 0; j < AtA_SIZE; j++) {
			corresp.AtA_sum[j] += corresp.AtA_host[i * AtA_SIZE + j];
		}
		for (int j = 0; j < Atb_SIZE; j++) {
			corresp.Atb_sum[j] += corresp.Atb_host[i * Atb_SIZE + j];
		}
	}
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < i; j++) {
			corresp.AtA_sum[i * 6 + j] = corresp.AtA_sum[j * 6 + i];
		}
	}

	TIME_SAMPLE_STOP("correspondence_GPU");

	if (ui_icp_gpu_draw) {
		copyFromDevice(correspondence_dev,
				corresp.points_dev, new_f.points_bn);

		cout << "GPU AtA_sum" << endl;
		for (int i = 0; i < 36; i++) {
			cout << corresp.AtA_sum[i] << ":";
		}
		cout << endl;
		for (int i = 0; i < 6; i++) {
			cout << corresp.Atb_sum[i] << ":";
		}
		cout << endl;
	}

	// put the resulting matrices into Eigen matrices
	// with row-major storage order
	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> AtA_eigen;
	Eigen::Matrix<double, 6, 1> Atb_eigen;

	// fill the matrices

	// use either CPU or GPU results
	for (int i = 0; i < 6; i++) {
		Atb_eigen(i) = corresp.Atb_sum[i];
		for (int j = 0; j < 6; j++) {
			AtA_eigen(i, j) = corresp.AtA_sum[i * 6 + j];
		}
	}

	double det = AtA_eigen.determinant();
	if (det == 0.0 || isnan(det)) {
		return -1.0;
	}

	Eigen::Matrix<float, 6, 1> parameters =
			AtA_eigen.llt().solve(Atb_eigen).cast<float>();

	// now get the resulting parameters back into a transformation matrix
	// set rotation from the angles
	t->resetTransform();

	for (int i = 0; i < 6; i++) {
		parameters(i) = -parameters(i);
	}

	t->setOrientation(ofQuaternion(ofRadToDeg(parameters(2)),
			ofVec3f(0.0, 0.0, 1.0),
			ofRadToDeg(parameters(1)), ofVec3f(0.0, 1.0, 0.0),
			ofRadToDeg(parameters(0)), ofVec3f(1.0, 0.0, 0.0)));
	t->setPosition(ofVec3f(-parameters(3), -parameters(4), -parameters(5)));

	return correspondence_ratio;
}


void ShapeApp::drawCorrespondence(Frame *frame, float *correspondence) {

	// now draw correspondence data with some lines
	// draw normals with lines
	ofPoint p, c;
	ofSetColor(ofColor::white);
	ofSetLineWidth(4.0);
	ofFill();

	int corresponding_n = 0;

	for (int y = 0; y < frame->host.height;
			y += frame->host.step) {
		for (int x = 0; x < frame->host.width;
				x += frame->host.step) {

			frame->host.getPoint(&p, x, y);
			getPoint(&c, correspondence, x, y, frame->host.width);

			if (c.x == 100.0 && c.y == 100.0) {
				ofSetColor(ofColor::yellow);
				ofCircle(p, c.z * 0.001);
			} else {
				if (c.length() > ui_distanse_threshold) {
					ofSetColor(ofColor::red);
				} else {
					ofSetColor(ofColor::white);
				}
				if (p.z != 0.0f && c.z != 0.0f) {
					ofLine(p, (p + c));
					corresponding_n++;
				}
			}

		}
	}

	cout << "drawCorrespondence corresponding_n " << corresponding_n << endl;
}
