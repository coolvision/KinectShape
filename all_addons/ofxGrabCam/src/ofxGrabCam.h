#pragma once
//
//  ofxGrabCam.h
//  ofxGrabCam
//
//  Created by Elliot Woods on 10/11/2011.
//	http://www.kimchiandchips.com
//

#include "ofMain.h"

//in normalised screen coords -0.5>0.5
#define OFXGRABCAM_SEARCH_WIDTH 1.0f
#define OFXGRABCAM_SEARCH_MAX_ITERATIONS 100
#define OFXGRABCAM_SEARCH_WINDINGS 10.0f
#define OFXGRABCAM_RESET_HOLD 500 //ms

class ofxGrabCam : public ofCamera {
public:	
	ofxGrabCam(bool useMouseListeners=true);
	~ofxGrabCam();
	
	void	begin(ofRectangle viewport = ofGetCurrentViewport());
	void	end(); ///< Overrides virtual end
	void	reset();
	
	const ofVec3f& getCursorWorld() { return mouseW; }
	const ofVec3f& getCursorProjected() { return mouseP; }
	
	void	setCursorWorld(const ofVec3f &);
	void	clearCursorWorld();
	
	void	setCursorDraw(bool enabled, float size=0.1);
	void	toggleCursorDraw();
	
	void	setMouseActions(bool enabled);
	void	toggleMouseActions();
	
	void	setFixUpwards(bool enabled);
	void	toggleFixUpwards();
	
	void	setTrackballRadius(float trackballRadius);
	float	getTrackballRadius() const;
	
	void	addListeners();
	void	removeListeners();
	
protected:
	
	////
	//events
	//
	void	update(ofEventArgs & args);
	void    mouseMoved(ofMouseEventArgs & args);
	void	mousePressed(ofMouseEventArgs & args);
	void	mouseReleased(ofMouseEventArgs & args);
	void	mouseDragged(ofMouseEventArgs & args);
	void	keyPressed(ofKeyEventArgs & args);
	void	keyReleased(ofKeyEventArgs & args);
	//
	bool	initialised;
	bool	mouseActions;
	//
	////
	
	
	////
	//cursor
	void	findCursor();
	//
	bool	mouseDown;
	bool	handDown;
	bool	altDown;
	bool	pickCursorFlag;
	ofVec3f mouseP;
	ofVec3f mouseW;
	bool	mouseWForced;
	bool	drawCursor;
	float	drawCursorSize;
	//
	////
	
	
	////
	//interactions
	int		resetDown;
	////
	
	////
	//view
	ofRectangle viewportRect;
	//
	GLint		viewport[4];
	GLdouble	matM[16], matP[16];
	//
	ofQuaternion rotation;
	//
	bool		fixUpwards;
	float		trackballRadius;
	//
	////
};