/*
 * ofxRangeToWorldCUDA.h
 *
 *  Created on: 2012
 *      Author: sk
 */

#pragma once

#include "Frame.h"

void sync();
void copyToDevice(void *to, void *from, size_t size, bool do_sync = true);
void copyFromDevice(void *to, void *from, size_t size, bool do_sync = true);

void rangeToWorld(CameraOptions *camera_opt, Frame *f);
