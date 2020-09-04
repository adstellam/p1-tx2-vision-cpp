#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <thread>
#include <ctime>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "libgpsmm.h"
#include "OBDII.h"
#include "OBDIICommunication.h"
#include "CabinViewFrameProcessor_Cascade.hpp"
#include "ForwardViewFrameProcessor_Baseline.hpp"

// const global variables defining file paths, gst pipelines, and frame size
const int FRAME_WIDTH = 480;
const int FRAME_HEIGHT = 360;
const std::string FACE_CASCADE_MODEL_PATH = "./data/haarcascade_frontalface_default.xml";
const std::string FACIAL_LANDMARK_MODEL_PATH = "./data/lbfmodel.yaml";
const std::string VEHICLE_CASCADE_MODEL_PATH = "./data/cars.xml";
const std::string CAB_CAM_CALIB_PARAMS_PATH = "./data/cab_cam_calib_params.yaml";
const std::string FWD_CAM_CALIB_PARAMS_PATH = "./data/fwd_cam_calib_params.yaml";
const std::string AUX_CAM_CALIB_PARAMS_PATH = "./data/aux_cam_calib_params.yaml";
const std::string CAB_VID_CAPTURE_GST_PIPELINE = "v4l2src device=/dev/video0 ! video/x-raw,format=UYVY,width=1280,height=720,framerate=(fraction)30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink";
const std::string FWD_VID_CAPTURE_GST_PIPELINE = "v4l2src device=/dev/video2 ! video/x-raw,format=UYVY,width=1280,height=720,framerate=(fraction)30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink";
const std::string AUX_VID_CAPTURE_GST_PIPELINE = "v4l2src device=/dev/video4 ! video/x-raw,format=UYVY,width=1280,height=720,framerate=(fraction)30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink";
const std::string CAB_VID_OUT_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)30/1 ! timeoverlay ! tee name=t t. ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=22211 t. ! queue x264enc ! h264parse ! queue ! mp4mux ! splitmuxsink location=log/vid/cab/v%04d.mp4 max-size-time=60000000";
const std::string FWD_VID_OUT_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)30/1 ! timeoverlay ! tee name=t t. ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=22212 t. ! queue x264enc ! h264parse ! queue ! mp4mux ! splitmuxsink location=log/vid/fwd/v%04d.mp4 max-size-time=60000000";
const std::string AUX_VID_OUT_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)30/1 ! timeoverlay ! tee name=t t. ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=22213 t. ! queue x264enc ! h264parse ! queue ! mp4mux ! splitmuxsink location=log/vid/aux/v%04d.mp4 max-size-time=60000000";
const std::string VEDAS_METRIC_OUT_PATH = "./log/dat/vedasmetric.dat";
/*
const std::string CAB_VID_OUT_PATH = "./log/vid/cab/v.api";
const std::string FWD_VID_OUT_PATH = "./log/vid/fwd/v.api";
const std::string AUX_VID_OUT_PATH = "./log/vid/aux/v.api";
const std::string CAB_VID_RTP_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)30/1 ! timeoverlay ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=22211";
const std::string FWD_VID_RTP_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)30/1 ! timeoverlay ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=22212";
const std::string AUX_VID_RTP_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)30/1 ! timeoverlay ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=22213";
*/
/*
const std::string CAB_VIN_PATH = "./data/cabin_vin.avi"; // for testing purpose only
const std::string FORWARD_VIEW_VIN_PATH = "./data/forward_vin.avi"; // for testing purpose only
*/

// non-const global variables
unsigned char cabflag;
unsigned char fwdflag; 
unsigned char gpsflag;
unsigned char obdflag;
std::vector<float> cabdata(4);
std::vector<float> fwddata(4);
std::vector<float> gpsdata(3);
std::vector<float> obddata(3);
std::stringstream vedas_metric_sstream;
cv::Mat cam_matrix_cab, dist_coeff_cab, cam_matrix_fwd, dist_coeff_fwd, cam_matrix_aux, dist_coeff_aux;

// function
void get_camera_calibration_parameters() {
	cv::Mat cam_matrix_double = cv::Mat::zeros(3, 3, CV_64FC1);
	cv::Mat dist_coeff_double = cv::Mat::zeros(5, 1, CV_64FC1);
	cv::FileStorage fs_c(CAB_CAM_CALIB_PARAMS_PATH, cv::FileStorage::READ);
	fs_c["camera_matrix"] >> cam_matrix_double;
	fs_c["distortion_coefficients"] >> dist_coeff_double; 
	cam_matrix_double.convertTo(cam_matrix_cab, CV_32FC1);
	dist_coeff_double.convertTo(dist_coeff_cab, CV_32FC1);
	fs_c.release();
	cv::FileStorage fs_f(FWD_CAM_CALIB_PARAMS_PATH, cv::FileStorage::READ);
	fs_f["camera_matrix"] >> cam_matrix_double;
	fs_f["distortion_coefficients"] >> dist_coeff_double; 
	cam_matrix_double.convertTo(cam_matrix_fwd, CV_32FC1);
	dist_coeff_double.convertTo(dist_coeff_fwd, CV_32FC1);
	fs_f.release();	
	cv::FileStorage fs_f(AUX_CAM_CALIB_PARAMS_PATH, cv::FileStorage::READ);
	fs_f["camera_matrix"] >> cam_matrix_double;
	fs_f["distortion_coefficients"] >> dist_coeff_double; 
	cam_matrix_double.convertTo(cam_matrix_aux, CV_32FC1);
	dist_coeff_double.convertTo(dist_coeff_aux, CV_32FC1);
	fs_f.release();	
}

// function
void process_cabin_view_frame() {
	cv::Ptr<CabinViewFrameProcessor_Cascade> cabin_view_frame_processor;
	const cv::Ptr<cv::CascadeClassifier> face_cascade = cv::Ptr<cv::CascadeClassifier>(new cv::CascadeClassifier());
	const cv::Ptr<cv::face::FacemarkLBF> facemark = cv::face::FacemarkLBF::create();
	face_cascade->load(FACE_CASCADE_MODEL_PATH);
	facemark->loadModel(FACIAL_LANDMARK_MODEL_PATH);
	cv::VideoCapture cap_c(CAB_VID_CAPTURE_GST_PIPELINE);
	if (!cap_c.isOpened()) {
		cabflag = 0x00;
		std::cout << "::main() >> Unable to open cabin video input" << std::endl;
	}
	cv::VideoWriter out_c(CAB_VID_OUT_GST_PIPELINE, 0, 20, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), true);
	/* 
	cv::VideoWriter out_c(CAB_VID_OUT_PATH, cv::VideoWriter::fourcc('M','J','P','G'), 20, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), true);
	cv::VideoWriter rtp_c(CAB_VID_OUT_RTP, cv::CAP_GSTREAMER, 0, 20, cv::Size(480, 360));
	*/
	if (!out_c.isOpened()) {
		std::cout << "::main() >> Unable to open file|use rtp for cabin video output" << std::endl;
	}
	cv::Mat frame, gray;
	while (cap_c.isOpened() & out_c.isOpened()) {
		if (cap_c.read(frame)) {
			cv::resize(frame, frame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), cv::INTER_LANCZOS4);
			cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
			cv::equalizeHist(gray, gray);
			cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(gray, face_cascade, facemark));
			cabin_view_frame_processor->analyze_frame();
			gray = cabin_view_frame_processor->get_annotated_frame();
			out_c.write(gray);
		}
	}
}

// function
void process_forward_view_frame() {
	cv::Ptr<ForwardViewFrameProcessor_Baseline> forward_view_frame_processor;
	const cv::Ptr<cv::CascadeClassifier> vehicle_cascade = cv::Ptr<cv::CascadeClassifier>(new cv::CascadeClassifier());
	const cv::Ptr<cv::HOGDescriptor> pedestrian_hog = cv::Ptr<cv::HOGDescriptor>(new cv::HOGDescriptor());
	vehicle_cascade->load(VEHICLE_CASCADE_MODEL_PATH);
	pedestrian_hog->setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	cv::VideoCapture cap_f(FWD_VID_CAPTURE_GST_PIPELINE);
	if (!cap_f.isOpened()) {
		fwdflag = 0x00;
		std::cout << "::main() >> Unable to open forward video input" << std::endl;
	}
	cv::VideoWriter out_f(FWD_VID_OUT_GST_PIPELINE, 0, 20, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), true);
	/*
	cv::VideoWriter out_f(FWD_VID_OUT_PATH, cv::VideoWriter::fourcc('M','J','P','G'), 20, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), true);
	cv::VideoWriter rtp_f(FWD_VID_OUT_RTP, cv::CAP_GSTREAMER, 0, 20, cv::Size(480, 360));
	*/
	if (!out_f.isOpened())
		std::cout << "::main() >> Unable to open file|use rtp for forward video output" << std::endl;
	cv::Mat frame, gray, hsv;
	while (cap_f.isOpened() & out_f.isOpened()) {
		if (cap_f.read(frame)) {
			cv::resize(frame, frame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), cv::INTER_LANCZOS4);
			cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
			cv::equalizeHist(gray, gray);
			forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(gray, hsv, vehicle_cascade, pedestrian_hog));
			forward_view_frame_processor->analyze_frame();
			gray = forward_view_frame_processor->get_annotated_frame();
			out_f.write(gray);
		}
	}
}

//function
void process_auxiliary_view_frame() {
	cv::VideoCapture cap_x(AUX_VID_CAPTURE_GST_PIPELINE);
	if (!cap_x.isOpened())
		std::cout << "::main() >> Unable to open auxiliary video input" << std::endl;
	cv::VideoWriter out_x(AUX_VID_OUT_GST_PIPELINE, 0, 20, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), true);
	/*
	cv::VideoWriter out_x(AUX_VID_OUT_PATH, cv::VideoWriter::fourcc('M','J','P','G'), 20, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), true);
	cv::VideoWriter rtp_x(AUX_VID_OUT_RTP, cv::CAP_GSTREAMER, 0, 20, cv::Size(480, 360));
	*/
	if (!out_x.isOpened())
		std::cout << "::main() >> Unable to open file|rtp for auxiliary video output" << std::endl;
	cv::Mat frame, gray;
	while (cap_x.isOpened() & out_x.isOpened()) {
		if (cap_x.read(frame)) {
			cv::resize(frame, frame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), cv::INTER_LANCZOS4);
			cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
			out_x.write(gray);
		}
	}
}

// function
void get_gps_data() {
	gpsmm gps_rec("localhost", DEFAULT_GPSD_PORT);
	if (gps_rec.stream(WATCH_ENABLE | WATCH_JSON) == NULL) {
		std::cout << "::main(): No GPS signal detected" << std::endl;
		gpsflag = 0x00;
		gpsdata[0] = 0; //latitude
		gpsdata[1] = 0; //longitude
		gpsdata[2] = 0; //speed
	}
	while (gps_rec.stream(WATCH_ENABLE | WATCH_JSON) != NULL) {
		if (!gps_rec.waiting(200000))
			continue;
		struct gps_data_t* gpsd_data;
		if ((gpsd_data = gps_rec.read()) == NULL) {
			std::cout << "::main(): Unable to read GPS data" << std::endl;
			break;
		} else {
			//timestamp_t ts {gpsd_data->fix.time};
			gpsflag = 0x07;
			gpsdata[0] = gpsd_data->fix.latitude;
			gpsdata[1] = gpsd_data->fix.longitude;
			gpsdata[2] = gpsd_data->fix.speed;
		}
	}
}

// function
void get_obd_data() {
	OBDIISocket s;
	if (OBDIIOpenSocket(&s, "can0", 0x7E0, 0x7E8, 0) < 0) {
		std::cout << "::main(): No OBD-II socket opened" << std::endl;
		obdflag = 0x00;
		obddata[0] = 0; //engineRPMs
		obddata[1] = 0; //distanceTraveledSinceCodesCleared
		obddata[2] = 0; //vehicleSpeed
	} else {
		bool engineRPMs_supported = false;
		bool distanceTraveledSinceCodesCleared_supported = false;
		bool vehicleSpeed_supported = false;
		OBDIICommandSet supported_commands = OBDIIGetSupportedCommands(&s);
		if (OBDIICommandSetContainsCommand(&supported_commands, OBDIICommands.engineRPMs))
			engineRPMs_supported = true;
		if (OBDIICommandSetContainsCommand(&supported_commands, OBDIICommands.distanceTraveledSinceCodesCleared))
			distanceTraveledSinceCodesCleared_supported = true;
		if (OBDIICommandSetContainsCommand(&supported_commands, OBDIICommands.vehicleSpeed))
			vehicleSpeed_supported = true;
		while (OBDIIOpenSocket(&s, "can0", 0x7E0, 0x7E8, 0) >= 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			obdflag = 0x00;
			if (engineRPMs_supported) {
				OBDIIResponse response = OBDIIPerformQuery(&s, OBDIICommands.engineRPMs);
				if (response.success) {
					obdflag |= 0x01;
					obddata[0] = response.numericValue;
				} else 
					std::cout << "::main(): Unable to read engineRPMs" << std::endl;
			}
			if (distanceTraveledSinceCodesCleared_supported) {
				OBDIIResponse response = OBDIIPerformQuery(&s, OBDIICommands.distanceTraveledSinceCodesCleared);
				if (response.success) {
					obdflag |= 0x02;
					obddata[1] = response.numericValue;
				} else
					std::cout << "::main(): Unable to read distanceTraveledSinceCodesCleared" << std::endl;
			}
			if (vehicleSpeed_supported) {
				OBDIIResponse response = OBDIIPerformQuery(&s, OBDIICommands.vehicleSpeed);
				if (response.success) {
					obdflag |= 0x04;
					obddata[2] = response.numericValue;
				} else
					std::cout << "::main(): Unable to read vehicleSpeed" << std::endl;
			}
		}
	}
}

// function 
void put_vedas_metric_in_stringstream() {
	while (true) {
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		vedas_metric_sstream 
			<< std::time(nullptr) << "\t"
		    << static_cast<int>(gpsflag) << "\t" 
		    << static_cast<int>(obdflag) << "\t" 
		    << static_cast<int>(cabflag) << "\t" 
		    << static_cast<int>(fwdflag) << "\t";
		vedas_metric_sstream << std::fixed; 
		for (int i = 0; i < gpsdata.size(); ++i) 
			vedas_metric_sstream << std::setprecision(4) << gpsdata[i] << "\t";
		for (int i = 0; i < obddata.size(); ++i)
			vedas_metric_sstream << std::setprecision(1) << obddata[i] << "\t";
		for (int i = 0; i < cabdata.size(); ++i)
			vedas_metric_sstream << std::setprecision(1) << cabdata[i] << "\t";
		for (int i = 0; i < fwddata.size() - 1 ; ++i)
			vedas_metric_sstream << std::setprecision(1) << fwddata[i] << "\t";
		vedas_metric_sstream << std::setprecision(1) << fwddata[fwddata.size() - 1] << "\n";
	}
}

// function
void flush_vedas_metric_sstream() {
	std::ofstream ofs;
	vedas_metric_sstream.str("");
	while (true) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		ofs.open(VEDAS_METRIC_OUT_PATH, std::ios::out);
		if (ofs.is_open()) {
			ofs << vedas_metric_sstream.str();
			vedas_metric_sstream.str("");
			ofs.close();
			std::cout << "::main() >> VeDas data written to the log file" << std::endl;
		} else
			std::cout << "::main() >> Unable to open VeDAS data log file: " << VEDAS_METRIC_OUT_PATH << std::endl;
	}
}

//
//
// main function
//
//
int main() {

	// get camera calibration parameters
	get_camera_calibration_parameters();	

	// create threads for tasks
	std::thread thd_cab(process_cabin_view_frame);
	thd_cab.detach();
	std::thread thd_fwd(process_forward_view_frame);
	thd_fwd.detach();
	std::thread thd_aux(process_auxiliary_view_frame);
	thd_aux.detach();
	std::thread thd_gps(get_gps_data);
	thd_gps.detach();
	std::thread thd_obd(get_obd_data);
	thd_obd.detach();
	std::thread thd_sst(put_vedas_metric_in_stringstream);
	thd_sst.detach();
	std::thread thd_log(flush_vedas_metric_sstream);
	thd_log.detach();
	
	//
	return 0;
}