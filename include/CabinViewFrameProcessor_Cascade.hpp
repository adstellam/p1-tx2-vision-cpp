#ifndef CabinViewFrameProcessor_Cascade_h_
#define CabinViewFrameProcessor_Cascade_h_

#include <iostream>
#include <sstream>
#include <string>
#include <array>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"

extern const int FRAME_WIDTH;
extern const int FRAME_HEIGHT;
extern cv::Mat cam_matrix_c, dist_coeff_c;
extern unsigned char cabflag;
extern std::vector<double> cabdata;

class CabinViewFrameProcessor_Cascade {
	private:
		//
		// CabinViewFrameProcessor_Cascade private member variables	
		//	
		cv::Mat _frame, _gray;
		cv::Ptr<cv::CascadeClassifier> _face_cascade;
		cv::Ptr<cv::face::Facemark> _facemark;
		std::vector<cv::Rect> _faces;
		std::vector<std::vector<cv::Point2f> > _shapes;
		cv::Rect _the_face;
		std::vector<cv::Point2f> _the_shape;
		std::vector<cv::Point2f> _the_eyes;
		std::vector<cv::Point2d> _projected_ref_pts;
		bool _ear_ready; // eye aspect ratio
		bool _tba_ready; // tate-bryan angles
		double _ear;
		double _pitch;
		double _yaw;
		double _roll;
		// 
		// CabinViewFrameProcessor_Cascade private member functions	
		//
		void compute_eye_aspect_ratio();
		void set_landmarks_opv(std::vector<cv::Point3d>& opv_lm);
		void set_landmarks_ipv(std::vector<cv::Point2d>& ipv_lm);
		void set_reference_opv(std::vector<cv::Point3d>& opv_rf);
		void estimate_head_pose();
		void set_cabflag_and_cabdata();
		void annotate_frame();
		//
	public:
		//
		// CabinViewFrameProcessor_Cascade public constructor
		//
		CabinViewFrameProcessor_Cascade(cv::Mat& frame, const cv::Ptr<cv::CascadeClassifier> face_cascade, const cv::Ptr<cv::face::Facemark> facemark);
		//
		// CabinViewFrameProcessor_Cascade public member variables	
		//
		void analyze_frame();
		cv::Mat& get_annotated_frame();
		//
};
#endif