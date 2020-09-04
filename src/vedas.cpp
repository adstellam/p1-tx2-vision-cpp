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

// const global variables defining frame size, file paths, and gst pipelines
extern const int FRAME_WIDTH = 480;
extern const int FRAME_HEIGHT = 360;
const std::string FACE_CASCADE_MODEL_PATH = "./data/haarcascade_frontalface_default.xml";
const std::string FACIAL_LANDMARK_MODEL_PATH = "./data/lbfmodel.yaml";
const std::string VEHICLE_CASCADE_MODEL_PATH = "./data/cars.xml";
const std::string CAB_CAM_CALIB_PARAMS_PATH = "./data/cab_cam_calib_params.yaml";
const std::string FWD_CAM_CALIB_PARAMS_PATH = "./data/fwd_cam_calib_params.yaml";
const std::string AUX_CAM_CALIB_PARAMS_PATH = "./data/aux_cam_calib_params.yaml";
const std::string CAB_VID_CAP_GST_PIPELINE = "v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=UYVY,width=1280,height=720,framerate=(fraction)30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink";
const std::string FWD_VID_CAP_GST_PIPELINE = "v4l2src device=/dev/video2 ! videoconvert ! video/x-raw,format=UYVY,width=1280,height=720,framerate=(fraction)30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink";
const std::string AUX_VID_CAP_GST_PIPELINE = "v4l2src device=/dev/video4 ! videoconvert ! video/x-raw,format=UYVY,width=1280,height=720,framerate=(fraction)30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink";
const std::string CAB_VID_STO_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)10/1 ! timeoverlay ! queue ! x264enc ! h264parse ! splitmuxsink location=log/vid/cab/v%04d.mp4 max-size-time=60000000000";
const std::string FWD_VID_STO_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)10/1 ! timeoverlay ! queue ! x264enc ! h264parse ! splitmuxsink location=log/vid/fwd/v%04d.mp4 max-size-time=60000000000";
const std::string AUX_VID_STO_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)10/1 ! timeoverlay ! queue ! x264enc ! h264parse ! splitmuxsink location=log/vid/aux/v%04d.mp4 max-size-time=60000000000";
const std::string CAB_VID_RTP_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)10/1 ! timeoverlay ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=25011";
const std::string FWD_VID_RTP_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)30/1 ! timeoverlay ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=25012";
const std::string AUX_VID_RTP_GST_PIPELINE = "appsrc ! videoconvert ! video/x-raw,format=I420,width=480,height=360,framerate=(fraction)30/1 ! timeoverlay ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=25013";
const std::string METRIC_FILE_PATH = "./log/dat/metric.dat";
/*
const std::string CAB_VIN_PATH = "./data/cabin_vin.avi"; // for testing purpose only
const std::string FWD_VIEW_VIN_PATH = "./data/forward_vin.avi"; // for testing purpose only
*/
/*
const std::string CAB_VID_OUT_PATH = "./log/vid/cab/v.api";
const std::string FWD_VID_OUT_PATH = "./log/vid/fwd/v.api";
const std::string AUX_VID_OUT_PATH = "./log/vid/aux/v.api";
*/

// non-const global variables
unsigned char cabflag;
unsigned char fwdflag; 
unsigned char gpsflag;
unsigned char obdflag;
std::vector<double> cabdata(4);
std::vector<double> fwddata(4);
std::vector<double> gpsdata(3);
std::vector<double> obddata(3);
std::stringstream vedas_metric_sstream;
cv::Mat cam_matrix_c, dist_coeff_c, cam_matrix_f, dist_coeff_f;

// function
void get_camera_calibration_parameters() {
	std::cout << "Reading camera calibration parameters . . . " << std::endl;
	//
	cam_matrix_c = cv::Mat(3, 3, CV_64FC1);
	dist_coeff_c = cv::Mat(5, 1, CV_64FC1);
	cv::FileStorage fs_c(CAB_CAM_CALIB_PARAMS_PATH, cv::FileStorage::READ);
	fs_c["camera_matrix"] >> cam_matrix_c;
	fs_c["distortion_coefficients"] >> dist_coeff_c; 
	fs_c.release();
	cam_matrix_f = cv::Mat(3, 3, CV_64FC1);
	dist_coeff_f = cv::Mat(5, 1, CV_64FC1);
	cv::FileStorage fs_f(FWD_CAM_CALIB_PARAMS_PATH, cv::FileStorage::READ);
	fs_f["camera_matrix"] >> cam_matrix_f;
	fs_f["distortion_coefficients"] >> dist_coeff_f; 
	fs_f.release();	
	//
	/*
	double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
	double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };
	cam_matrix_c = cv::Mat(3, 3, CV_64FC1, K);
    dist_coeff_c = cv::Mat(5, 1, CV_64FC1, D);
    cam_matrix_f = cv::Mat(3, 3, CV_64FC1, K);
    dist_coeff_f = cv::Mat(5, 1, CV_64FC1, D);
    */
}

// function
void analyze_cabin_view_frame(cv::Ptr<CabinViewFrameProcessor_Cascade> cabin_view_frame_processor) {
	cabin_view_frame_processor->analyze_frame();
}

// function
void analyze_forward_view_frame(cv::Ptr<ForwardViewFrameProcessor_Baseline> forward_view_frame_processor) {
	forward_view_frame_processor->analyze_frame();
}

//function
void process_auxiliary_view_frame(cv::Mat& frame) {
	cv::resize(frame, frame, cv::Size(FRAME_WIDTH, FRAME_HEIGHT), cv::INTER_LANCZOS4);
	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
}

// function
void get_gps_data() {
	std::cout << "Starting to get gps data . . . " << std::endl;
	gpsmm gps_rec("localhost", DEFAULT_GPSD_PORT);
	if (gps_rec.stream(WATCH_ENABLE | WATCH_JSON) == NULL) {
		std::cout << "No GPS signal detected" << std::endl;
		gpsflag = 0x00;
		gpsdata[0] = 0; //latitude
		gpsdata[1] = 0; //longitude
		gpsdata[2] = 0; //speed
	}
	while (gps_rec.stream(WATCH_ENABLE | WATCH_JSON) != NULL) {
		if (!gps_rec.waiting(200)) { //milliseconds
			gpsflag = 0x00;
			continue;
		}
		struct gps_data_t* gpsd_data;
		if ((gpsd_data = gps_rec.read()) == NULL) {
			std::cout << "Unable to read GPS data" << std::endl;
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
	std::cout << "Starting to get obdii data . . . " << std::endl;
	OBDIISocket s;
	if (OBDIIOpenSocket(&s, "can0", 0x7E0, 0x7E8, 0) < 0) {
		std::cout << "No OBD-II socket detected" << std::endl;
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
					std::cout << "Unable to read engineRPMs" << std::endl;
			}
			if (distanceTraveledSinceCodesCleared_supported) {
				OBDIIResponse response = OBDIIPerformQuery(&s, OBDIICommands.distanceTraveledSinceCodesCleared);
				if (response.success) {
					obdflag |= 0x02;
					obddata[1] = response.numericValue;
				} else
					std::cout << "Unable to read distanceTraveledSinceCodesCleared" << std::endl;
			}
			if (vehicleSpeed_supported) {
				OBDIIResponse response = OBDIIPerformQuery(&s, OBDIICommands.vehicleSpeed);
				if (response.success) {
					obdflag |= 0x04;
					obddata[2] = response.numericValue;
				} else
					std::cout << "Unable to read vehicleSpeed" << std::endl;
			}
		}
	}
}

// function 
void prepare_vedas_metric_sstream() {
	std::cout << "Preparing vedas metric sstream at every 200 millisec . . ." << std::endl;
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
	std::cout << "Flushing vedas metric sstream at every 1 sec . . ." << std::endl;
	std::ofstream ofs;
	vedas_metric_sstream.str("");
	while (true) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		ofs.open(METRIC_FILE_PATH, std::ios::app);
		if (ofs.is_open()) {
			ofs << vedas_metric_sstream.str();
			vedas_metric_sstream.str("");
			ofs.close();
		} else
			std::cout << "Unable to open VeDAS metric log file: " << METRIC_FILE_PATH << std::endl;
	}
}

//
//
// main function
//
//
int main() {

	// launching threads for background tasks
	std::thread th_gps(get_gps_data);
	th_gps.detach();
	std::thread th_obd(get_obd_data);
	th_obd.detach();
	std::thread th_sst(prepare_vedas_metric_sstream);
	th_sst.detach();
	std::thread th_log(flush_vedas_metric_sstream);
	th_log.detach();
	
	// loading camera calibration parameters
	get_camera_calibration_parameters();	

	// instantiating video frame processors and object detection models
	cv::Ptr<CabinViewFrameProcessor_Cascade> cabin_view_frame_processor;
	cv::Ptr<ForwardViewFrameProcessor_Baseline> forward_view_frame_processor;
	const cv::Ptr<cv::CascadeClassifier> face_cascade = cv::Ptr<cv::CascadeClassifier>(new cv::CascadeClassifier());
	const cv::Ptr<cv::CascadeClassifier> vehicle_cascade = cv::Ptr<cv::CascadeClassifier>(new cv::CascadeClassifier());
	const cv::Ptr<cv::face::FacemarkLBF> facemark = cv::face::FacemarkLBF::create();
	const cv::Ptr<cv::HOGDescriptor> pedestrian_hog = cv::Ptr<cv::HOGDescriptor>(new cv::HOGDescriptor());
	face_cascade->load(FACE_CASCADE_MODEL_PATH);
	vehicle_cascade->load(VEHICLE_CASCADE_MODEL_PATH);
	facemark->loadModel(FACIAL_LANDMARK_MODEL_PATH);
	pedestrian_hog->setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	// creating cv::VideoCapture instances
	cv::VideoCapture cap_c(CAB_VID_CAP_GST_PIPELINE);
	if (!cap_c.isOpened()) {
		cabflag = 0x00;
		std::cout << "Unable to open cabin video input" << std::endl;
	}
	cv::VideoCapture cap_f(FWD_VID_CAP_GST_PIPELINE);
	if (!cap_f.isOpened()) {
		fwdflag = 0x00;
		std::cout << "Unable to open forward video input" << std::endl;
	}
	cv::VideoCapture cap_x(AUX_VID_CAP_GST_PIPELINE);
	if (!cap_x.isOpened())
		std::cout << "Unable to open auxiliary video input" << std::endl;

	// creating cv::VideoWriter instances
	cv::VideoWriter sto_c(CAB_VID_STO_GST_PIPELINE, cv::CAP_GSTREAMER, 0, 10, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
	cv::VideoWriter rtp_c(CAB_VID_RTP_GST_PIPELINE, cv::CAP_GSTREAMER, 0, 10, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
	cv::VideoWriter sto_f(FWD_VID_STO_GST_PIPELINE, cv::CAP_GSTREAMER, 0, 10, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
	cv::VideoWriter rtp_f(FWD_VID_RTP_GST_PIPELINE, cv::CAP_GSTREAMER, 0, 10, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
	cv::VideoWriter sto_x(AUX_VID_STO_GST_PIPELINE, cv::CAP_GSTREAMER, 0, 10, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
	cv::VideoWriter rtp_x(AUX_VID_RTP_GST_PIPELINE, cv::CAP_GSTREAMER, 0, 10, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
	
	// checking whether cv::VideoCapture|cv::VideoWriter is ready for each video input|output
	unsigned char vio = 0x00;
	if (cap_c.isOpened() && sto_c.isOpened()) {
		vio |= 0x01;
		std::cout << "Starting to process cabin-view video frames . . . " << std::endl;
	}
	if (cap_f.isOpened() && sto_c.isOpened()) {
		vio |= 0x02;
		std::cout << "Starting to process forward-view video frames . . . " << std::endl;
	}
	if (cap_x.isOpened() && sto_c.isOpened()) {
		vio |= 0x04;
		std::cout << "Starting to process auxiliary-view video frames . . . " << std::endl;
	}

	// declaring variables for video frames and performance measures
	cv::Mat frame_c, frame_f, frame_x;
	int64 t0, t1, t2, t3, t4;
			
	// Processing video frames
	bool c, f, x;
	switch (vio) {
		
		case 0x00:
			std::cout << "None of video input|output ready" << std::endl;
			break;

		case 0x01:
			while (cap_c.read(frame_c)) {
				t0 = cv::getTickCount();
				cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(frame_c, face_cascade, facemark));
				cabin_view_frame_processor->analyze_frame();
				frame_c = cabin_view_frame_processor->get_annotated_frame();
				t1 = cv::getTickCount();
				std::cout << " * " << (t1-t0)/cv::getTickFrequency()*1000 << "|";
				sto_c.write(frame_c);
				t2 = cv::getTickCount();
				std::cout << (t2-t1)/cv::getTickFrequency()*1000 << "|";
				rtp_c.write(frame_c);
				t3 = cv::getTickCount();
				std::cout << (t3-t2)/cv::getTickFrequency()*1000 << "|";
				cv::imshow("Cabin View", frame_c);
				cv::waitKey(1);
				t4 = cv::getTickCount();
				std::cout << (t4-t3)/cv::getTickFrequency()*1000 << " ms" << std::endl;;
			}
			break;

		case 0x02:
			while (cap_f.read(frame_f)) {
				t0 = cv::getTickCount();
				forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(frame_f, vehicle_cascade, pedestrian_hog));
				forward_view_frame_processor->analyze_frame();
				frame_f = forward_view_frame_processor->get_annotated_frame();
				t1 = cv::getTickCount();
				std::cout << " * " << (t1-t0)/cv::getTickFrequency()*1000 << "|";
				sto_f.write(frame_f);
				t2 = cv::getTickCount();
				std::cout << (t2-t1)/cv::getTickFrequency()*1000 << "|";
				rtp_f.write(frame_f);
				t3 = cv::getTickCount();
				std::cout << (t3-t2)/cv::getTickFrequency()*1000 << "|";
				cv::imshow("Forward View", frame_f);
				cv::waitKey(1);
				t4 = cv::getTickCount();
				std::cout << (t4-t3)/cv::getTickFrequency()*1000 << " ms" << std::endl;;
			}
			break;

		case 0x04:
			while (cap_x.read(frame_x)) {
				process_auxiliary_view_frame(frame_x);
				sto_x.write(frame_x);
				rtp_x.write(frame_x);
			}
			break;

		case 0x03:
			c = cap_c.read(frame_c);
			f = cap_f.read(frame_f);
			while (c && f) {
				t0 = cv::getTickCount();
				cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(frame_c, face_cascade, facemark));
				forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(frame_f, vehicle_cascade, pedestrian_hog));
				std::thread th_cab(analyze_cabin_view_frame, cabin_view_frame_processor);
				std::thread th_fwd(analyze_forward_view_frame, forward_view_frame_processor);
				th_cab.join();
				frame_c = cabin_view_frame_processor->get_annotated_frame();
				sto_c.write(frame_c);
				rtp_c.write(frame_c);
				c = cap_c.read(frame_c);
				th_fwd.join();
				t0 = cv::getTickCount();
				frame_f = forward_view_frame_processor->get_annotated_frame();
				sto_f.write(frame_f);
				rtp_f.write(frame_f);
				f = cap_f.read(frame_f);
				t1 = cv::getTickCount();
				std::cout << "Processing cycle for both cabin/forward view frames: " << (t1-t0)/cv::getTickFrequency()*1000 << " ms" << std::endl;
			}
			while (c && !f) {
				cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(frame_c, face_cascade, facemark));
				cabin_view_frame_processor->analyze_frame();
				frame_c = cabin_view_frame_processor->get_annotated_frame();
				sto_c.write(frame_c);
				rtp_c.write(frame_c);
				c = cap_c.read(frame_c);
			}
			while (!c && f) {
				forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(frame_f, vehicle_cascade, pedestrian_hog));
				forward_view_frame_processor->analyze_frame();
				frame_f = forward_view_frame_processor->get_annotated_frame();
				sto_f.write(frame_f);
				rtp_f.write(frame_f);
				f = cap_f.read(frame_f);
			}
			break;

		case 0x05:
			c = cap_c.read(frame_c);
			x = cap_x.read(frame_x);
			while (c && x) {
				cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(frame_c, face_cascade, facemark));
				std::thread th_cab(analyze_cabin_view_frame, cabin_view_frame_processor);
				std::thread th_aux(process_auxiliary_view_frame, std::ref(frame_x));
				th_aux.join();
				sto_x.write(frame_x);
				rtp_x.write(frame_x);
				x = cap_x.read(frame_x);
				th_cab.join();
				frame_c = cabin_view_frame_processor->get_annotated_frame();
				sto_c.write(frame_c);
				rtp_c.write(frame_c);
				c = cap_c.read(frame_c);
			}
			while (c && !x) {
				cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(frame_c, face_cascade, facemark));
				cabin_view_frame_processor->analyze_frame();
				frame_c = cabin_view_frame_processor->get_annotated_frame();
				sto_c.write(frame_c);
				rtp_c.write(frame_c);
				c = cap_c.read(frame_c);
			}
			while (!c && x) {
				process_auxiliary_view_frame(frame_x);
				sto_x.write(frame_x);
				rtp_x.write(frame_x);
				x = cap_x.read(frame_x);
			}
			break;

		case 0x06:
			f = cap_f.read(frame_f);
			x = cap_x.read(frame_x);
			while (f && x) {
				forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(frame_f, vehicle_cascade, pedestrian_hog));
				std::thread th_fwd(analyze_forward_view_frame, forward_view_frame_processor);
				std::thread th_aux(process_auxiliary_view_frame, std::ref(frame_x));
				th_aux.join();
				sto_x.write(frame_x);
				rtp_x.write(frame_x);
				x = cap_x.read(frame_x);
				th_fwd.join();
				frame_f = forward_view_frame_processor->get_annotated_frame();
				sto_f.write(frame_f);
				rtp_f.write(frame_f);
				f = cap_f.read(frame_f);
			}
			while (f && !x) {
				forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(frame_f, vehicle_cascade, pedestrian_hog));
				forward_view_frame_processor->analyze_frame();
				frame_f = forward_view_frame_processor->get_annotated_frame();
				sto_f.write(frame_f);
				rtp_f.write(frame_f);
				f = cap_f.read(frame_f);
			}
			while (!f && x) {
				process_auxiliary_view_frame(frame_x);
				sto_x.write(frame_x);
				rtp_x.write(frame_x);
				x = cap_x.read(frame_x);
			}
			break;

		case 0x07:
			c = cap_c.read(frame_c);
			f = cap_f.read(frame_f);
			x = cap_x.read(frame_x);
			while (c && f && x) {
				cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(frame_c, face_cascade, facemark));
				forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(frame_f, vehicle_cascade, pedestrian_hog));
				std::thread th_cab(analyze_cabin_view_frame, cabin_view_frame_processor);
				std::thread th_fwd(analyze_forward_view_frame, forward_view_frame_processor);
				std::thread th_aux(process_auxiliary_view_frame, std::ref(frame_x));
				th_aux.join();
				sto_x.write(frame_x);
				rtp_x.write(frame_x);
				x = cap_x.read(frame_x);
				th_cab.join();
				frame_c = cabin_view_frame_processor->get_annotated_frame();
				sto_c.write(frame_c);
				rtp_c.write(frame_c);
				c = cap_c.read(frame_c);
				th_fwd.join();
				frame_f = forward_view_frame_processor->get_annotated_frame();
				sto_f.write(frame_f);
				rtp_f.write(frame_f);
				f = cap_f.read(frame_f);
			}
			while (c && f && !x) {
				cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(frame_c, face_cascade, facemark));
				forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(frame_f, vehicle_cascade, pedestrian_hog));
				std::thread th_cab(analyze_cabin_view_frame, cabin_view_frame_processor);
				std::thread th_fwd(analyze_forward_view_frame, forward_view_frame_processor);
				th_cab.join();
				frame_c = cabin_view_frame_processor->get_annotated_frame();
				sto_c.write(frame_c);
				rtp_c.write(frame_c);
				c = cap_c.read(frame_c);
				th_fwd.join();
				frame_f = forward_view_frame_processor->get_annotated_frame();
				sto_f.write(frame_f);
				rtp_f.write(frame_f);
				f = cap_f.read(frame_f);
			}
			while (c && !f && x) {
				cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(frame_c, face_cascade, facemark));
				std::thread th_cab(analyze_cabin_view_frame, cabin_view_frame_processor);
				std::thread th_aux(process_auxiliary_view_frame, std::ref(frame_x));
				th_aux.join();
				sto_x.write(frame_x);
				rtp_x.write(frame_x);
				x = cap_x.read(frame_x);
				th_cab.join();
				frame_c = cabin_view_frame_processor->get_annotated_frame();
				sto_c.write(frame_c);
				rtp_c.write(frame_c);
				c = cap_c.read(frame_c);
			}
			while (!c && f && x) {
				forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(frame_f, vehicle_cascade, pedestrian_hog));
				std::thread th_fwd(analyze_forward_view_frame, forward_view_frame_processor);
				std::thread th_aux(process_auxiliary_view_frame, std::ref(frame_x));
				th_aux.join();
				sto_x.write(frame_x);
				rtp_x.write(frame_x);
				x = cap_x.read(frame_x);
				th_fwd.join();
				frame_f = forward_view_frame_processor->get_annotated_frame();
				sto_f.write(frame_f);
				rtp_f.write(frame_f);
				f = cap_f.read(frame_f);
			}
			while (c && !f && !x) {
				cabin_view_frame_processor = cv::Ptr<CabinViewFrameProcessor_Cascade>(new CabinViewFrameProcessor_Cascade(frame_c, face_cascade, facemark));
				cabin_view_frame_processor->analyze_frame();
				frame_c = cabin_view_frame_processor->get_annotated_frame();
				sto_c.write(frame_c);
				rtp_c.write(frame_c);
				c = cap_c.read(frame_c);
			}
			while (!c && f && !x) {
				forward_view_frame_processor = cv::Ptr<ForwardViewFrameProcessor_Baseline>(new ForwardViewFrameProcessor_Baseline(frame_f, vehicle_cascade, pedestrian_hog));
				forward_view_frame_processor->analyze_frame();
				frame_f = forward_view_frame_processor->get_annotated_frame();
				sto_f.write(frame_f);
				rtp_f.write(frame_f);
				f = cap_f.read(frame_f);
			}
			while (!c && !f && x) {
				process_auxiliary_view_frame(frame_x);
				sto_x.write(frame_x);
				rtp_x.write(frame_x);
				x = cap_x.read(frame_x);
			}
			break;

		default:
			std::cout << "None of video input|output available" << std::endl;
	}

	//
	return 0;
}
/**********************************************************************************
  To demonstrate the streaming of video without the rtsp server, use the following
  GST pipeline with gst-launch-1.0:

  gst-launch-1.0 udpsrc port=25011 caps='application/x-rtp,media=video,clock-rate=90000,encoding=RAW,sampling=YCbCr-4:2:0,depth=(string)8,width=(string)480,height=(string)360,payload=96' ! queue ! rtpvrawdepay ! videoconvert ! autovideosink

***********************************************************************************/