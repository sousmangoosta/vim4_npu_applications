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

static const char *names[] = {
	"face"
};

void* init_retinaface_network_file(const char *mpath) {

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

int set_retinaface_input(void *qcontext, cv::Mat orig_img, int input_width, int input_high, int input_channel) {

	int ret = 0;
	int input_size = 0;
	int hw = input_width*input_high;
	unsigned char *rawdata = NULL;
	cv::Mat temp_img(input_width, input_high, CV_8UC3), normalized_img, img;
	if (orig_img.cols >= orig_img.rows)
  	{
  		int y_padding = orig_img.cols - orig_img.rows;
  		cv::copyMakeBorder(orig_img, img, 0, y_padding, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  		cv::resize(img, temp_img, cv::Size(input_width, input_high));
  	}
  	else
  	{
  		int x_padding = orig_img.rows - orig_img.cols;
  		cv::copyMakeBorder(orig_img, img, 0, 0, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  		cv::resize(img, temp_img, cv::Size(input_width, input_high));
  	}
  	
	cv::cvtColor(temp_img, temp_img, cv::COLOR_BGR2RGB);
	temp_img.convertTo(normalized_img, CV_32FC3, 1.0 / 255.0);

	rawdata = normalized_img.data;
	nn_input inData;
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

int run_retinaface_network(void *qcontext, face_detect_out_t &retinaface_detect_out) {

	int ret = 0;
	nn_output *outdata = NULL;
	aml_output_config_t outconfig;
	memset(&outconfig, 0, sizeof(aml_output_config_t));

	outconfig.format = AML_OUTDATA_FLOAT32;//AML_OUTDATA_RAW or AML_OUTDATA_FLOAT32
	outconfig.typeSize = sizeof(aml_output_config_t);
	outconfig.order = AML_OUTPUT_ORDER_NCHW;

	outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);
	if (outdata == NULL) {
		printf("aml_module_output_get error\n");
		return -1;
	}
	
	postprocess_retinaface(outdata, &retinaface_detect_out);
    return ret;
}



int destroy_retinaface_network(void *qcontext) {

	int ret = aml_module_destroy(qcontext);
	return ret;
}


