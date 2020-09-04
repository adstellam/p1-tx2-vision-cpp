#include "ForwardViewFrameProcessor_Baseline.hpp"

//
//
/*********** PRIVATE MEMBER FUNCTIONS *************/
//	
//

// ForwardViewFrameProcessor_Baseline private member function		
void ForwardViewFrameProcessor_Baseline::prepare_lane_coords_for_fitting(const std::vector<cv::Vec4i>& hough_lines, std::vector<double>& llane_coord_x, std::vector<double>& llane_coord_y, std::vector<double>& rlane_coord_x, std::vector<double>& rlane_coord_y) {
	llane_coord_x.clear();
	llane_coord_y.clear();
	rlane_coord_x.clear();
	rlane_coord_y.clear();
	double x1, y1, x2, y2, slope;
	for (int i = 0; i < hough_lines.size(); ++i) {
		x1 = static_cast<double>(hough_lines[i][0]);
		y1 = static_cast<double>(hough_lines[i][1]);
		x2 = static_cast<double>(hough_lines[i][2]);
		y2 = static_cast<double>(hough_lines[i][3]);
		if (x1 != x2) {
			slope = (y2-y1)/(x2-x1);
			if (slope < -0.2) {
				llane_coord_x.push_back(x1);
				llane_coord_y.push_back(y1);
				llane_coord_x.push_back(x2);
				llane_coord_y.push_back(y2);
			}
			if (slope > 0.2) {
				rlane_coord_x.push_back(x1);
				rlane_coord_y.push_back(y1);
				rlane_coord_x.push_back(x2);
				rlane_coord_y.push_back(y2);								
			}
		} 
	}
}

// ForwardViewFrameProcessor_Baseline private member function
bool ForwardViewFrameProcessor_Baseline::linear_fit(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& coef) {
	bool fitted = false;
	coef.clear();
	if (x.size() >= 2 && x.size() == y.size()) {
		cv::Mat X = cv::Mat::ones(x.size(), 2, CV_64FC1);
		for (int i =0; i < x.size(); ++i)
			X.at<double>(i,1) = x[i];
		cv::Mat Y = cv::Mat(y);
		Y.convertTo(Y, CV_64FC1);
		cv::Mat B = cv::Mat(2, 1, CV_64FC1);
		fitted = cv::solve(X, Y, B, cv::DECOMP_NORMAL);
		coef.push_back(B.at<double>(0, 0));
		coef.push_back(B.at<double>(1, 0));
	} 
	return fitted;
}

// ForwardViewFrameProcessor_Baseline private member function
bool ForwardViewFrameProcessor_Baseline::quadratic_fit(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& coef) {
	bool fitted = false;
	coef.clear();
	if (x.size() > 2  && x.size() == y.size()) {
		cv::Mat X = cv::Mat::ones(x.size(), 3, CV_64FC1);
		for (int j = 0; j < 3; ++j)
			for (int i = 0; i < x.size(); ++i)
				X.at<double>(i, j) = std::pow(x[i], j);		
				cv::Mat Y = cv::Mat(y);
		Y.convertTo(Y, CV_64FC1);
		cv::Mat B = cv::Mat(3, 1, CV_64FC1);
		fitted = cv::solve(X, Y, B, cv::DECOMP_NORMAL);
		coef.push_back(B.at<double>(0, 0));
		coef.push_back(B.at<double>(1, 0));
		coef.push_back(B.at<double>(2, 0));
	} 
	return fitted;
}	

// ForwardViewFrameProcessor_Baseline private member function
double ForwardViewFrameProcessor_Baseline::polyval(const double x, const std::vector<double>& coef) {
	double val = 0.0;
	for (int j = 0; j < coef.size(); ++j) {
		val += std::pow(x, j) * coef[j];
	}
	return val;
}

// ForwardViewFrameProcessor_Baseline private member function		
void ForwardViewFrameProcessor_Baseline::fit_lanes(const std::vector<double>& llane_coord_x, const std::vector<double>& llane_coord_y, const std::vector<double>& rlane_coord_x, const std::vector<double>& rlane_coord_y) {
	std::vector<double> llane_coef, rlane_coef;
	int u1, v1, u2, v2;
	_llane.detected = false;
	_rlane.detected = false;
	if (llane_coord_x.size() >= 2) {
		bool fitted = linear_fit(llane_coord_y, llane_coord_x, llane_coef);
		if (fitted) {
			v1 = _gray.rows;
			u1 = static_cast<int>(polyval(v1*1.0, llane_coef));
			v2 = static_cast<int>(_gray.rows*0.7);
			u2 = static_cast<int>(polyval(v2*1.0, llane_coef));
			_llane.detected = true;
			_llane.endpoints = cv::Vec4i(u1, v1, u2, v2);
			_llane.offset = (_gray.cols/2.0 - u1)/_gray.cols;
		}
	}
	if (rlane_coord_x.size() >= 2) {
		bool fitted = linear_fit(rlane_coord_y, rlane_coord_x, rlane_coef);
		if (fitted) {
			v1 = _gray.rows;
			u1 = static_cast<int>(polyval(v1*1.0, rlane_coef));
			v2 = static_cast<int>(_gray.rows*0.7);
			u2 = static_cast<int>(polyval(v2*1.0, rlane_coef));
			_rlane.detected = true;
			_rlane.endpoints = cv::Vec4i(u1, v1, u2, v2);
			_rlane.offset = (u1 - _gray.cols/2.0)/_gray.cols;
		} 
	}
}

// ForwardViewFrameProcessor_Baseline private member function		
void ForwardViewFrameProcessor_Baseline::compute_lane_alignment() {	
	//
	cv::Mat mask_white = cv::Mat::zeros(_gray.size(), CV_8UC1);
	cv::Mat mask_yellow = cv::Mat::zeros(_gray.size(), CV_8UC1);
	cv::Mat mask_white_or_yellow = cv::Mat::zeros(_gray.size(), CV_8UC1); 
	cv::Mat mask_polygon = cv::Mat::zeros(_gray.size(), CV_8UC1);
	cv::Mat gray_masked, gray_blurred, canny_edged, gray_for_hough;
	std::vector<cv::Vec4i> hough_lines;
	std::vector<double> llane_coord_x, llane_coord_y, rlane_coord_x, rlane_coord_y;
	//
	const double rho = 2.0;
	const double theta = CV_PI/180.0;
	const int thresh = 25;
	const double min_line_length = 50.0;
	const double max_line_gap = 200.0;
	//
	cv::inRange(_gray, 200, 255, mask_white);
	cv::inRange(_hsv, cv::Scalar(20, 100, 100), cv::Scalar(30, 255, 255), mask_yellow);
	cv::bitwise_or(mask_white, mask_yellow, mask_white_or_yellow);
	_gray.copyTo(gray_masked, mask_white_or_yellow);
	cv::GaussianBlur(gray_masked, gray_blurred, cv::Size(5, 5), 0, 0);
	cv::Canny(gray_blurred, canny_edged, 40, 120);
	//
	cv::Point vertices[4];
	enum PolygonVertices {BOTTOM_LEFT, TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT};
	vertices[BOTTOM_LEFT] = cv::Point(static_cast<int>(_gray.cols*0.1), _gray.rows);
	vertices[TOP_LEFT] = cv::Point(static_cast<int>(_gray.cols*0.35), static_cast<int>(_gray.rows*0.5));
	vertices[TOP_RIGHT] = cv::Point(static_cast<int>(_gray.cols*0.65), static_cast<int>(_gray.rows*0.5));
	vertices[BOTTOM_RIGHT] = cv::Point(static_cast<int>(_gray.cols*0.9), _gray.rows);					
	cv::fillConvexPoly(mask_polygon, vertices, 4, cv::Scalar(255));
	canny_edged.copyTo(gray_for_hough, mask_polygon);
	cv::HoughLinesP(gray_for_hough, hough_lines, rho, theta, thresh, min_line_length, max_line_gap);
	//
	prepare_lane_coords_for_fitting(hough_lines, llane_coord_x, llane_coord_y, rlane_coord_x, rlane_coord_y);
	fit_lanes(llane_coord_x, llane_coord_y, rlane_coord_x, rlane_coord_y);
}

// ForwardViewFrameProcessor_Baseline private member function
void ForwardViewFrameProcessor_Baseline::detect_vehicles() {
	DetectedVehicleData v;
	cv::Point2f bbox_ctr, bbox_ctr_trans;
	std::vector<cv::Rect> dets;
	_vehicle_cascade->detectMultiScale(_gray, dets, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(40,40));
	_vehicles.clear();
	if (dets.size() > 0) 
		for (int i = 0; i < dets.size(); ++i) {
			v.bbox = dets[i];
			bbox_ctr = cv::Point2f(dets[i].x + dets[i].width/2.0, dets[i].y + dets[i].height/2.0);
			bbox_ctr_trans = cv::Point2f(bbox_ctr.x - _gray.cols/2.0, _gray.rows - bbox_ctr.y);
			// direction is the angle from the straight forward line, ranging from -90 (L) to +90 (R).  
			v.direction = 90.0 - std::atan2(bbox_ctr_trans.y, bbox_ctr_trans.x)*180.0/CV_PI;
			// distance as calculated here is a very rough estimate. 
			v.distance = 2.0/dets[i].width * _gray.cols; 
			if (std::abs(v.direction) <= 45)
				v.relevant = true;
			else
				v.relevant = false;
			_vehicles.push_back(v);
		}
}

// ForwardViewFrameProcessor_Baseline private function
void ForwardViewFrameProcessor_Baseline::detect_pedestrians() {
	DetectedPedestrianData p;
	cv::Point2f bbox_ctr, bbox_ctr_trans;
	std::vector<cv::Rect> dets;
	_pedestrian_hog->detectMultiScale(_gray, dets, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2); 
	_pedestrians.clear();
	if (dets.size() > 0) 
		for (int i = 0; i < dets.size(); ++i) {
			p.bbox = dets[i];
			p.bbox.x += static_cast<int>(p.bbox.x * 0.1);
			p.bbox.y += static_cast<int>(p.bbox.x * 0.1);
			p.bbox.width = static_cast<int>(p.bbox.width * 0.8);
			p.bbox.height = static_cast<int>(p.bbox.height * 0.8);
			bbox_ctr = cv::Point2f(dets[i].x + dets[i].width/2.0, dets[i].y + dets[i].height/2.0);
			bbox_ctr_trans = cv::Point2f(bbox_ctr.x - _gray.cols/2.0, _gray.rows - bbox_ctr.y);
			// direction is the angle from the straight forward line, ranging from -90 (L) to +90 (R).   
			p.direction = static_cast<int>(90.0 - std::atan2(bbox_ctr_trans.y, bbox_ctr_trans.x)*180.0/CV_PI);
			if (std::abs(p.direction) <= 45)
				p.relevant = true;
			else
				p.relevant = false;
			_pedestrians.push_back(p);
		}
}

// ForwardViewFrameProcessor_Baseline private member function
void ForwardViewFrameProcessor_Baseline::set_fwdflag_and_fwddata() {
	fwdflag = 0x00;
	if (_llane.detected) {
		fwdflag |= 0x01;
		fwddata[0] = _llane.offset;
	} 
	if (_rlane.detected) {
		fwdflag |= 0x02;
		fwddata[1] = _rlane.offset;
	} 
	if (_vehicles.size() > 0) {
		double min_distance = std::numeric_limits<double>::max();
		double direction_for_min_distance = 0;
		for (int i = 0; i < _vehicles.size(); ++i) {
			if (_vehicles[i].relevant && _vehicles[i].distance < min_distance) {
				min_distance = _vehicles[i].distance;
				direction_for_min_distance = _vehicles[i].direction;
			}
		}
		fwdflag |= 0x0C;
		fwddata[2] = min_distance;
		fwddata[3] = direction_for_min_distance;
	}
}

// ForwardViewFrameProcessor_Baseline private member function
void ForwardViewFrameProcessor_Baseline::annotate_frame() {
	if (_llane.detected) {
		cv::Point pt1 = cv::Point(_llane.endpoints[0], _llane.endpoints[1]);
		cv::Point pt2 = cv::Point(_llane.endpoints[2], _llane.endpoints[3]);
		cv::line(_frame, pt1, pt2, cv::Scalar(255, 0, 0), 2); 
	}
	if (_rlane.detected) {
		cv::Point pt1 = cv::Point(_rlane.endpoints[0], _rlane.endpoints[1]);
		cv::Point pt2 = cv::Point(_rlane.endpoints[2], _rlane.endpoints[3]);
		cv::line(_frame, pt1, pt2, cv::Scalar(255, 0, 0), 2); 
	}
	cv::line(_frame, cv::Point(static_cast<int>(_frame.cols/2), _frame.rows), cv::Point(static_cast<int>(_frame.cols/2), static_cast<int>(_frame.rows*0.8)), cv::Scalar(255, 255, 0), 2);
	int baseline = 0;
	cv::String loffsetStr, roffsetStr;
	cv::Point loffsetLoc, roffsetLoc;
	cv::Size loffsetTextSize, roffsetTextSize;
	if (_llane.detected) 
		loffsetStr = std::to_string(_llane.offset*100);
	else
		loffsetStr = "undefined";
	if (_rlane.detected)
		roffsetStr = std::to_string(_rlane.offset*100);
	else
		roffsetStr = "undefined";
	loffsetTextSize = cv::getTextSize(loffsetStr, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
	loffsetLoc = cv::Point(static_cast<int>(_frame.cols/2 - loffsetTextSize.width - 30), static_cast<int>(_frame.rows*0.9));	
	roffsetLoc = cv::Point(static_cast<int>(_frame.cols/2 + 10), static_cast<int>(_frame.rows*0.9));	
	cv::putText(_frame, loffsetStr, loffsetLoc, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 127, 255), 1);
	cv::putText(_frame, roffsetStr, roffsetLoc, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 127, 255), 1);	

	cv::String vdescStr, pdescStr;
	cv::Point vdescLoc, pdescLoc;
	cv::Size vdescTextSize, pdescTextSize;
	for (int i = 0; i < _vehicles.size(); ++i) {
		cv::rectangle(_frame, _vehicles[i].bbox, cv::Scalar(0, 0, 255), 2);
		if (_vehicles[i].relevant) 
			vdescStr = std::to_string(_vehicles[i].direction) + "|" + std::to_string(_vehicles[i].distance);
		else
			vdescStr = std::to_string(_vehicles[i].direction) + "|ignored";			
 		vdescTextSize = cv::getTextSize(vdescStr, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
 		vdescLoc = cv::Point(_vehicles[i].bbox.x, _vehicles[i].bbox.y - vdescTextSize.height - 10);
		cv::putText(_frame, vdescStr, vdescLoc, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 127, 255), 1);
	}
	/*
	for (int i = 0; i < _pedestrians.size(); ++i) {
		cv::rectangle(_frame, _pedestrians[i].bbox, cv::Scalar(0, 0, 255), 2);
		if (_pedestrians[i].relevant)
			pdescStr = std::to_string(_pedestrians[i].direction);
		else
			pdescStr = std::to_string(_pedestrians[i].direction) + "|ignored";			
 		pdescTextSize = cv::getTextSize(pdescStr, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
 		pdescLoc = cv::Point(_pedestrians[i].bbox.x, _pedestrians[i].bbox.y - pdescTextSize.height - 10);
		cv::putText(_frame, pdescStr, pdescLoc, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 127, 255), 1);
	}
	*/
}

//
//
/*********** PUBLIC CONSTRUCTOR *************/
//
//
ForwardViewFrameProcessor_Baseline::ForwardViewFrameProcessor_Baseline(cv::Mat& frame, const cv::Ptr<cv::CascadeClassifier>& vehicle_cascade, const cv::Ptr<cv::HOGDescriptor>& pedestrian_hog) {
	cv::resize(frame, _frame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), cv::INTER_LANCZOS4);
	cv::cvtColor(_frame, _gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(_frame, _hsv, cv::COLOR_BGR2HSV);
	cv::equalizeHist(_gray, _gray);
	_vehicle_cascade = vehicle_cascade;
	_pedestrian_hog = pedestrian_hog;	
}

//
//
/*********** PUBLIC MEMBER FUNCTIONS *************/
//
//

// ForwardViewFrameProcessor_Baseline public member function
void ForwardViewFrameProcessor_Baseline::analyze_frame() {
	compute_lane_alignment();
	if (_llane.detected) 
		std::cout << std::fixed << std::setprecision(1) << "LLO [" << _llane.offset*100 << "] ";
	else
		std::cout << "LLO [N/A] "; 
	if (_rlane.detected)
		std::cout << std::fixed << std::setprecision(1) << "RLO [" << _rlane.offset*100 << "] ";
	else 
		std::cout << "RLO [N/A] "; 
	detect_vehicles();
	std::cout << "Vehs [" << _vehicles.size() << "]" << std::endl;
	/* 
	detect_pedestrians();
	std::cout << "Peds [" << _pedestrians.size() << "]" << std::endl;
	*/
	set_fwdflag_and_fwddata();
}

// ForwardViewFrameProcessor_Baseline public member function
cv::Mat& ForwardViewFrameProcessor_Baseline::get_annotated_frame() {
	annotate_frame();
	return _frame;
}
