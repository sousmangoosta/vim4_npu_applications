/****************************************************************************
*
*    Copyright (c) 2019  by amlogic Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of amlogic Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of amlogic Corporation.
*
***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn_sdk.h"
#include "nn_util.h"
#include "postprocess_util.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <getopt.h>

#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640

struct option longopts[] = {
	{ "path",           required_argument,  NULL,   'p' },
	{ "model",          required_argument,  NULL,   'M' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};


nn_input inData;

cv::Mat img;

aml_module_t modelType;

static int input_width,input_high,input_channel;

static const char *names[] = {
	"face"
};

typedef enum _amlnn_detect_type_ {
	Accuracy_Detect_Yolo_V3 = 0
} amlnn_detect_type;



void* init_network_file(const char *mpath) {

	void *qcontext = NULL;

	aml_config config;
	memset(&config, 0, sizeof(aml_config));

	config.nbgType = NN_ADLA_FILE;
	config.path = mpath;
	config.modelType = ADLA_LOADABLE;
	config.typeSize = sizeof(aml_config);

	qcontext = aml_module_create(&config);
	if (qcontext == NULL) {
		printf("amlnn_init is fail\n");
		return NULL;
	}

	if (config.nbgType == NN_ADLA_MEMORY && config.pdata != NULL) {
		free((void*)config.pdata);
	}

	return qcontext;
}

int set_input(void *qcontext, const char *jpath) {

	int ret = 0;
	int input_size = 0;
	int hw = input_width*input_high;
	unsigned char *rawdata = NULL;
	cv::Mat temp_img(MODEL_WIDTH, MODEL_HEIGHT, CV_8UC3),normalized_img;
	img = cv::imread(jpath);
	cv::cvtColor(img, temp_img, cv::COLOR_BGR2RGB);
	cv::resize(temp_img, temp_img, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
	temp_img.convertTo(normalized_img, CV_32FC3, 1.0 / 255.0);

	rawdata = normalized_img.data;
	inData.input_type = BINARY_RAW_DATA;
	inData.input = rawdata;
	inData.input_index = 0;
	inData.size = input_width * input_high * input_channel * sizeof(float);
	ret = aml_module_input_set(qcontext, &inData);

	return ret;
}

static cv::Scalar obj_id_to_color(int obj_id) {

	int const colors[6][3] = { { 1,0,1 }, { 0,0,1 }, { 0,1,1 }, { 0,1,0 }, { 1,1,0 }, { 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}

int run_network(void *qcontext) {

	int ret = 0;
	nn_output *outdata = NULL;
	aml_output_config_t outconfig;
	memset(&outconfig, 0, sizeof(aml_output_config_t));

	outconfig.format = AML_OUTDATA_FLOAT32;//AML_OUTDATA_RAW or AML_OUTDATA_FLOAT32
	outconfig.typeSize = sizeof(aml_output_config_t);
	outconfig.order = AML_OUTPUT_ORDER_NCHW;

	face_detect_out_t retinaface_detect_out;
	outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);
	if (outdata == NULL) {
		printf("aml_module_output_get error\n");
		return -1;
	}
	
	postprocess_retinaface(outdata, &retinaface_detect_out);

	int classid = 0;
	float prob = 0;
	int left = 0, right = 0, top = 0, bot = 0;

	cv::Point pt1;
	cv::Point pt2;

	int baseline;

	cv::namedWindow("Image Window");

	printf("face_num:%d\n", retinaface_detect_out.detNum);

	for (int i =0;i < retinaface_detect_out.detNum;i++){
		classid = (int)retinaface_detect_out.pBox[i].objectClass;
		prob = retinaface_detect_out.pBox[i].score;
        
		left  = (retinaface_detect_out.pBox[i].x - retinaface_detect_out.pBox[i].w/2.) * img.cols;
		right = (retinaface_detect_out.pBox[i].x + retinaface_detect_out.pBox[i].w/2.) * img.cols;
		top   = (retinaface_detect_out.pBox[i].y - retinaface_detect_out.pBox[i].h/2.) * img.rows;
		bot   = (retinaface_detect_out.pBox[i].y + retinaface_detect_out.pBox[i].h/2.) * img.rows;
		if (left < 2) left = 2;
		if (right > img.cols-2) right = img.cols-2;
		if (top < 2) top = 2;
		if (bot > img.rows-2) bot = img.rows-2;
		printf("class:%s,label_num:%d,prob:%f,left:%d,top:%d,right:%d,bot:%d\n",names[classid], classid, prob, left, top, right, bot);

		pt1=cv::Point(left, top);
		pt2=cv::Point(right, bot);
		cv::Rect rect(left, top, right-left, bot-top);
		cv::rectangle(img, rect, obj_id_to_color(classid), 1, 8, 0);
		
		cv::Point point1(retinaface_detect_out.point_1[i].x * img.cols, retinaface_detect_out.point_1[i].y * img.rows);
		cv::Point point2(retinaface_detect_out.point_2[i].x * img.cols, retinaface_detect_out.point_2[i].y * img.rows);
		cv::Point point3(retinaface_detect_out.point_3[i].x * img.cols, retinaface_detect_out.point_3[i].y * img.rows);
		cv::Point point4(retinaface_detect_out.point_4[i].x * img.cols, retinaface_detect_out.point_4[i].y * img.rows);
		cv::Point point5(retinaface_detect_out.point_5[i].x * img.cols, retinaface_detect_out.point_5[i].y * img.rows);
		
		cv::circle(img, point1, 2, cv::Scalar(255, 0, 0), -1);
		cv::circle(img, point2, 2, cv::Scalar(0, 255, 0), -1);
		cv::circle(img, point3, 2, cv::Scalar(0, 0, 255), -1);
		cv::circle(img, point4, 2, cv::Scalar(0, 255, 255), -1);
		cv::circle(img, point5, 2, cv::Scalar(255, 0, 255), -1);

		if (top < 50) {
			top = 50;
			left +=10;
		}
		cv::Size text_size = cv::getTextSize(names[classid], cv::FONT_HERSHEY_COMPLEX, 0.5 , 1, &baseline);
		cv::Rect rect1(left, top-20, text_size.width+10, 20);
		cv::rectangle(img, rect1, obj_id_to_color(classid), -1);
		cv::putText(img, names[classid], cvPoint(left+5,top-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0), 1);
	}
	cv::imwrite("output.bmp", img);
	cv::imshow("Image Window",img);
	cv::waitKey(0);

    return ret;
}

int destroy_network(void *qcontext) {

	int ret = aml_module_destroy(qcontext);
	return ret;
}

int main(int argc,char **argv)
{
	int c;
	int ret = 0;
	void *context = NULL;
	char *model_path = NULL;
	char *input_data = NULL;
	input_width = MODEL_WIDTH;
	input_high = MODEL_HEIGHT;
	input_channel = 3;

	while ((c = getopt_long(argc, argv, "p:m:H", longopts, NULL)) != -1) {
		switch (c) {
			case 'p':
				input_data = optarg;
				break;

			case 'm':
				model_path = optarg;
				break;

			default:
				printf("%s [-p picture path] [-m model path]  [-H]\n", argv[0]);
				exit(1);
		}
	}

	context = init_network_file(model_path);
	if (context == NULL) {
		printf("init_network fail.\n");
		return -1;
	}

	ret = set_input(context, input_data);

	if (ret != 0) {

		printf("set_input fail.\n");
		return -1;
	}

	ret = run_network(context);

	if (ret != 0) {
		printf("run_network fail.\n");
		return -1;
	}

	ret = destroy_network(context);

	if (ret != 0) {
		printf("destroy_network fail.\n");
		return -1;
	}

	return ret;
}
