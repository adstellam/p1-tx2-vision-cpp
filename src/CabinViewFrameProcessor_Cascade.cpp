#include "CabinViewFrameProcessor_Cascade.hpp"

//
//
/*********** PRIVATE MEMBER FUNCTIONS *************/
//
//

// CabinViewFrameProcessor_Cascade private member function 
void CabinViewFrameProcessor_Cascade::compute_eye_aspect_ratio() {
	for (int i = 0; i < 12; ++i) {
		_the_eyes[i].x = _the_shape[36 + i].x;
		_the_eyes[i].y = _the_shape[36 + i].y;
	}
	double v_dist_r1 = std::sqrt(std::pow((_the_eyes[1].x-_the_eyes[5].x), 2) + std::pow((_the_eyes[1].y-_the_eyes[5].y), 2));
	double v_dist_r2 = std::sqrt(std::pow((_the_eyes[2].x-_the_eyes[4].x), 2) + std::pow((_the_eyes[2].y-_the_eyes[4].y), 2));
	double v_dist_l1 = std::sqrt(std::pow((_the_eyes[7].x-_the_eyes[11].x), 2) + std::pow((_the_eyes[7].y-_the_eyes[11].y), 2));
	double v_dist_l2 = std::sqrt(std::pow((_the_eyes[8].x-_the_eyes[10].x), 2) + std::pow((_the_eyes[8].y-_the_eyes[10].y), 2));
	double h_dist_r = std::sqrt(std::pow((_the_eyes[0].x-_the_eyes[3].x), 2) + std::pow((_the_eyes[0].y-_the_eyes[3].y), 2));
	double h_dist_l = std::sqrt(std::pow((_the_eyes[6].x-_the_eyes[9].x), 2) + std::pow((_the_eyes[6].y-_the_eyes[9].y), 2));
	double ear_r = (v_dist_r1 + v_dist_r2)/(2 * h_dist_r);
	double ear_l = (v_dist_l1 + v_dist_l2)/(2 * h_dist_l);
	_ear_ready = true;
	_ear = (ear_r + ear_l)/2.0;
}

// CabinViewFrameProcessor_Cascade private member function
void CabinViewFrameProcessor_Cascade::set_landmarks_opv(std::vector<cv::Point3d>& opv_lm) {
	opv_lm.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
    opv_lm.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
    opv_lm.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
    opv_lm.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
    opv_lm.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
    opv_lm.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
    opv_lm.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
    opv_lm.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
    opv_lm.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
    opv_lm.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
    opv_lm.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
    opv_lm.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
    opv_lm.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
    opv_lm.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner
}

// CabinViewFrameProcessor_Cascade private member function
void CabinViewFrameProcessor_Cascade::set_landmarks_ipv(std::vector<cv::Point2d>& ipv_lm) {
	ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[17])); //#17 left brow left corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[21])); //#21 left brow right corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[22])); //#22 right brow left corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[26])); //#26 right brow right corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[36])); //#36 left eye left corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[39])); //#39 left eye right corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[42])); //#42 right eye left corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[45])); //#45 right eye right corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[31])); //#31 nose left corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[35])); //#35 nose right corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[48])); //#48 mouth left corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[54])); //#54 mouth right corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[57])); //#57 mouth central bottom corner
    ipv_lm.push_back(static_cast<cv::Point2d>(_the_shape[8]));  //#8 chin corner
}

// CabinViewFrameProcessor_Cascade private member function
void CabinViewFrameProcessor_Cascade::set_reference_opv(std::vector<cv::Point3d>& opv_rf) {
	opv_rf.push_back(cv::Point3d(10.0, 10.0, 10.0));
    opv_rf.push_back(cv::Point3d(10.0, 10.0, -10.0));
    opv_rf.push_back(cv::Point3d(10.0, -10.0, -10.0));
    opv_rf.push_back(cv::Point3d(10.0, -10.0, 10.0));
    opv_rf.push_back(cv::Point3d(-10.0, 10.0, 10.0));
    opv_rf.push_back(cv::Point3d(-10.0, 10.0, -10.0));
    opv_rf.push_back(cv::Point3d(-10.0, -10.0, -10.0));
    opv_rf.push_back(cv::Point3d(-10.0, -10.0, 10.0));
}

// CabinViewFrameProcessor_Cascade private member function
void CabinViewFrameProcessor_Cascade::estimate_head_pose() {
	
	cv::Mat rvec = cv::Mat(3, 1, CV_64FC1); // rodrigues vector 
	cv::Mat tvec = cv::Mat(3, 1, CV_64FC1); // translation vector t
	cv::Mat rmat = cv::Mat(3, 3, CV_64FC1); // rotation matrix R
	cv::Mat pmat = cv::Mat(3, 4, CV_64FC1); // pose matrix [R|t]
	cv::Mat avec = cv::Mat(3, 1, CV_64FC1); // tait-bryant|eurler angle vector
	cv::Mat tvec_out = cv::Mat(3, 1, CV_64FC1);
    cv::Mat rmat_out = cv::Mat(3, 3, CV_64FC1);
    cv::Mat intr_out = cv::Mat(3, 3, CV_64FC1);
	std::vector<cv::Point3d> opv_lm;
	std::vector<cv::Point2d> ipv_lm;
	std::vector<cv::Point3d> opv_rf;
	std::vector<cv::Point2d> ipv_rf;
	set_landmarks_opv(opv_lm);
	set_landmarks_ipv(ipv_lm);
	cv::solvePnP(opv_lm, ipv_lm, cam_matrix_c, dist_coeff_c, rvec, tvec);
	cv::Rodrigues(rvec, rmat);
	cv::hconcat(rmat, tvec, pmat);
    cv::decomposeProjectionMatrix(pmat, intr_out, rmat_out, tvec_out, cv::noArray(), cv::noArray(), cv::noArray(), avec);
    set_reference_opv(opv_rf);
	cv::projectPoints(opv_rf, rvec, tvec, cam_matrix_c, dist_coeff_c, ipv_rf);
	_tba_ready = true;
	_pitch = avec.at<double>(0);
	_yaw = avec.at<double>(1);
	_roll = avec.at<double>(2);
	_projected_ref_pts = ipv_rf;
}

// CabinViewFrameProcessor_Cascade private member function
void CabinViewFrameProcessor_Cascade::set_cabflag_and_cabdata() {
	cabflag = 0x00;
	if (_ear_ready) {
		cabflag |= 0x01;
		cabdata[0] = _ear*100;
	} 
	if (_tba_ready) {
		cabflag |= 0x0E;
		cabdata[1] = _pitch;
		cabdata[2] = _yaw;
		cabdata[3] = _roll;
	}
}

// CabinViewFrameProcessor_Cascade private member function
void CabinViewFrameProcessor_Cascade::annotate_frame() {
	std::stringstream text_content;
	int baseline = 0;
	cv::Size text_size = cv::getTextSize("Eye", cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
	if (_faces.size() > 0) {
		cv::rectangle(_frame, _the_face, cv::Scalar(255, 0, 0), 2);	
		if (_ear_ready) {
			cv::line(_frame, cv::Point(static_cast<int>(_the_eyes[0].x), static_cast<int>(_the_eyes[0].y)), cv::Point(static_cast<int>(_the_eyes[3].x), static_cast<int>(_the_eyes[3].y)), cv::Scalar(255, 0 ,0), 2);
			cv::line(_frame, cv::Point(static_cast<int>(_the_eyes[1].x), static_cast<int>(_the_eyes[1].y)), cv::Point(static_cast<int>(_the_eyes[5].x), static_cast<int>(_the_eyes[5].y)), cv::Scalar(255, 0 ,0), 2);
			cv::line(_frame, cv::Point(static_cast<int>(_the_eyes[2].x), static_cast<int>(_the_eyes[2].y)), cv::Point(static_cast<int>(_the_eyes[4].x), static_cast<int>(_the_eyes[4].y)), cv::Scalar(255, 0 ,0), 2);
			cv::line(_frame, cv::Point(static_cast<int>(_the_eyes[6].x), static_cast<int>(_the_eyes[6].y)), cv::Point(static_cast<int>(_the_eyes[9].x), static_cast<int>(_the_eyes[9].y)), cv::Scalar(255, 0 ,0), 2);
			cv::line(_frame, cv::Point(static_cast<int>(_the_eyes[7].x), static_cast<int>(_the_eyes[7].y)), cv::Point(static_cast<int>(_the_eyes[11].x), static_cast<int>(_the_eyes[11].y)), cv::Scalar(255, 0 ,0), 2);
			cv::line(_frame, cv::Point(static_cast<int>(_the_eyes[8].x), static_cast<int>(_the_eyes[8].y)), cv::Point(static_cast<int>(_the_eyes[10].x), static_cast<int>(_the_eyes[10].y)), cv::Scalar(255, 0 ,0), 2);
			text_content.str("");
			text_content << "Eye aspect ratio: " << std::fixed << std::setprecision(1) << _ear*100;
			cv::putText(_frame, text_content.str(), cv::Point(15, text_size.height*2.0), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 127, 255), 2);
		} else {
			cv::putText(_frame, "Unable to fit shape", cv::Point(15, text_size.height*2.0), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);			
		}		
	} else {
		cv::putText(_frame, "No face detected", cv::Point(15, text_size.height*2.0), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
	}
	if (_tba_ready) {
		text_content.str("");
		text_content << "Pitch|Yaw|Roll: " << std::fixed << std::setprecision(1) << _pitch << " " << _yaw << " " << _roll;
		cv::putText(_frame, text_content.str(), cv::Point(15, text_size.height*3.6), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 127, 255), 2);	
		cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[0].x), static_cast<int>(_projected_ref_pts[0].y)), cv::Point(static_cast<int>(_projected_ref_pts[1].x), static_cast<int>(_projected_ref_pts[1].y)), cv::Scalar(0, 255, 0));
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[1].x), static_cast<int>(_projected_ref_pts[1].y)), cv::Point(static_cast<int>(_projected_ref_pts[2].x), static_cast<int>(_projected_ref_pts[2].y)), cv::Scalar(0, 255, 0));
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[2].x), static_cast<int>(_projected_ref_pts[2].y)), cv::Point(static_cast<int>(_projected_ref_pts[3].x), static_cast<int>(_projected_ref_pts[3].y)), cv::Scalar(0, 255, 0));
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[4].x), static_cast<int>(_projected_ref_pts[4].y)), cv::Point(static_cast<int>(_projected_ref_pts[5].x), static_cast<int>(_projected_ref_pts[5].y)), cv::Scalar(0, 255, 0));
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[5].x), static_cast<int>(_projected_ref_pts[5].y)), cv::Point(static_cast<int>(_projected_ref_pts[6].x), static_cast<int>(_projected_ref_pts[6].y)), cv::Scalar(0, 255, 0));
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[6].x), static_cast<int>(_projected_ref_pts[6].y)), cv::Point(static_cast<int>(_projected_ref_pts[7].x), static_cast<int>(_projected_ref_pts[7].y)), cv::Scalar(0, 255, 0));
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[1].x), static_cast<int>(_projected_ref_pts[1].y)), cv::Point(static_cast<int>(_projected_ref_pts[5].x), static_cast<int>(_projected_ref_pts[5].y)), cv::Scalar(0, 255, 0));
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[2].x), static_cast<int>(_projected_ref_pts[2].y)), cv::Point(static_cast<int>(_projected_ref_pts[6].x), static_cast<int>(_projected_ref_pts[6].y)), cv::Scalar(0, 255, 0));
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[7].x), static_cast<int>(_projected_ref_pts[7].y)), cv::Point(static_cast<int>(_projected_ref_pts[4].x), static_cast<int>(_projected_ref_pts[4].y)), cv::Scalar(0, 0, 255), 2);
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[0].x), static_cast<int>(_projected_ref_pts[0].y)), cv::Point(static_cast<int>(_projected_ref_pts[4].x), static_cast<int>(_projected_ref_pts[4].y)), cv::Scalar(0, 0, 255), 2);
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[3].x), static_cast<int>(_projected_ref_pts[3].y)), cv::Point(static_cast<int>(_projected_ref_pts[0].x), static_cast<int>(_projected_ref_pts[0].y)), cv::Scalar(0, 0, 255), 2);
	    cv::line(_frame, cv::Point(static_cast<int>(_projected_ref_pts[3].x), static_cast<int>(_projected_ref_pts[3].y)), cv::Point(static_cast<int>(_projected_ref_pts[7].x), static_cast<int>(_projected_ref_pts[7].y)), cv::Scalar(0, 0, 255), 2);
	}
}

//
//
/*********** PUBLIC CONSTRUCTOR *************/
//
//
CabinViewFrameProcessor_Cascade::CabinViewFrameProcessor_Cascade(cv::Mat& frame, const cv::Ptr<cv::CascadeClassifier> face_cascade, const cv::Ptr<cv::face::Facemark> facemark) {
	cv::resize(frame, _frame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), cv::INTER_LANCZOS4);
	cv::cvtColor(_frame, _gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(_gray, _gray);
	_face_cascade = face_cascade;
	_facemark = facemark;
	_the_shape = std::vector<cv::Point2f>(64);
	_the_eyes = std::vector<cv::Point2f>(12);
}

//
//
/*********** PUBLIC MEMBER FUNCTIONS *************/
//
//

// CabinViewFrameProcessor_Cascade public member function
void CabinViewFrameProcessor_Cascade::analyze_frame() {
	_ear_ready = false;
	_tba_ready = false;
	_face_cascade->detectMultiScale(_gray, _faces, 1.1, 2.0, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30,30)); 
	if (_faces.size() > 0) {
		std::cout << "Face detected " << _faces.size() << ": ";
		int face_area;
		int max_face_area = 0;
		int argmax_face_area = 0;
		for (int i = 0; i < _faces.size(); ++i) {
			face_area = _faces[i].width * _faces[i].height;
			if (face_area > max_face_area) {
				max_face_area = face_area;
				argmax_face_area = i;
			}
		}
		_the_face = _faces[argmax_face_area];
		bool fitted = _facemark->fit(_gray, _faces, _shapes);
		if (fitted) {
			_the_shape = _shapes[argmax_face_area];
			compute_eye_aspect_ratio();
			estimate_head_pose();
			if (_ear_ready)
				if (_tba_ready)
					std::cout << std::fixed << std::setprecision(1) << "EAR [" << _ear*100 << "] TBA [" << _pitch << ", " << _yaw << ", " << _roll << "]";
				else 
					std::cout << std::fixed << std::setprecision(1) << "EAR [" << _ear*100 << "] TBA [N/A]"; 
			else 
				if (_tba_ready)
					std::cout << std::fixed << std::setprecision(1) << "EAR [N/A" << "] TBA [" << _pitch << ", " << _yaw << ", " << _roll << "]";
				else
					std::cout << "EAR [N/A] TBA[N/A]";
		}
	} else {
		std::cout << "No face detected";
	}
	set_cabflag_and_cabdata();
}

// CabinViewFrameProcessor_Cascade public member function
cv::Mat& CabinViewFrameProcessor_Cascade::get_annotated_frame() {
	annotate_frame();
	return _frame;
}

