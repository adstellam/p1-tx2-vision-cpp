#ifndef ForwardViewFrameProcessor_Baseline_h_
#define ForwardViewFrameProcessor_Baseline_h_

#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <cmath>
#include <limits>
#include "opencv2/opencv.hpp"

extern const int FRAME_WIDTH;
extern const int FRAME_HEIGHT;
extern cv::Mat cam_matrix_f, dist_coeff_f;
extern unsigned char fwdflag;
extern std::vector<double> fwddata;

struct LaneAlignmentData {
	bool detected;
	cv::Vec4i endpoints;
	double offset;
};
struct DetectedVehicleData {
	bool relevant;
	cv::Rect bbox;
	double direction;
	double distance;
};
struct DetectedPedestrianData {
	bool relevant;
	cv::Rect bbox;
	double direction;
};
		
class ForwardViewFrameProcessor_Baseline {
	private:
		//
		// ForwardViewFrameProcessor_Baseline private member variables
		//
		cv::Mat _frame, _gray, _hsv;
		cv::Ptr<cv::CascadeClassifier> _vehicle_cascade;
		cv::Ptr<cv::HOGDescriptor> _pedestrian_hog;
		LaneAlignmentData _llane, _rlane;
		std::vector<DetectedVehicleData> _vehicles;
		std::vector<DetectedPedestrianData> _pedestrians;
		
		//
		// ForwardViewFrameProcessor_Baseline private member functions
		//
		void prepare_lane_coords_for_fitting(const std::vector<cv::Vec4i>& hough_lines, std::vector<double>& llane_coord_x, std::vector<double>& llane_coord_y, std::vector<double>& rlane_coord_x, std::vector<double>& rlane_coord_y);
		bool linear_fit(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& coef);
		bool quadratic_fit(const std::vector<double>& x, const std::vector<double>& y, std::vector<double>& coef);
		double polyval(const double x, const std::vector<double>& coef);
		void fit_lanes(const std::vector<double>& llane_coord_x, const std::vector<double>& llane_coord_y, const std::vector<double>& rlane_coord_x, const std::vector<double>& rlane_coord_y);
		void compute_lane_alignment();	
		void detect_vehicles();
		void detect_pedestrians();
		void set_fwdflag_and_fwddata();
		void annotate_frame();

	public:
		// 
		// ForwardViewFrameProcessor_Baseline public constructor
		//
		ForwardViewFrameProcessor_Baseline(cv::Mat& frame, const cv::Ptr<cv::CascadeClassifier>& vehicle_cascade, const cv::Ptr<cv::HOGDescriptor>& pedestrian_hog);
		// 
		// ForwardViewFrame Processor_Baseline public member functions
		//
		void analyze_frame();
		cv::Mat& get_annotated_frame();
};
#endif