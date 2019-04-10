// tf_tiny_yoloV3.cpp : 定义 DLL 应用程序的导出函数。
//

#include "stdafx.h"
#include "iostream"
#include "string.h"
#include "stdlib.h"

#define TF_TINY_YOLOV3_EXPORTS
#include "tf_tiny_yoloV3.h"

#include <vector>
#include <eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <list>
#include <armadillo>

#include "opencv2/opencv.hpp"

#include "TestTensorflow.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/ops/array_ops.h"

#pragma warning(disable:4996)

using namespace arma;
using namespace tensorflow;
using namespace tensorflow::ops;

void get_class(std::string path, std::vector<std::string> *class_names)
{
	std::ifstream infile;
	infile.open(path.data());
	assert(infile.is_open());

	std::string class_name;
	while (std::getline(infile, class_name))
	{
		class_names->push_back(class_name);
	}
}

mat get_anchors_mask(int num_layers)
{
	mat anchor_mask;
	if (num_layers == 3)
		anchor_mask = "6 7 8; 3 4 5; 0 1 2;";
	if (num_layers == 2)
		anchor_mask = "3 4 5; 1 2 3;";
	return anchor_mask;
}

mat get_anchors(int num_layers)
{
	mat anchors;
	if (num_layers == 3)
		anchors = "10 13; 16 30; 33 23; 30 61; 62 45; 59 119; 116 90; 156 198; 373 326;";
	if (num_layers == 2)
	{
		anchors = "10 14; 23 27; 37 58; 81 82; 135 169; 344 319;";
	}
	return anchors;
}

double sigmoid(float x)
{
	double y = 1.0 / (1.0 + exp(-x));
	return y;
}

void CVMat_to_Tensor(cv::Mat img, Tensor* output_tensor, int input_rows, int input_cols)
{
	img.convertTo(img, CV_32FC3);
	img = img / 255.0;

	float *p = output_tensor->flat<float>().data();

	cv::Mat tempMat(input_rows, input_cols, CV_32FC3, p);
	img.convertTo(tempMat, CV_32FC3);
}

void letterbox_image(cv::Mat imgIn, int input_height, int input_width, cv::Mat *imgOut)
{
	float iw = imgIn.cols;
	float ih = imgIn.rows;
	float scale = MIN(input_width / iw, input_height / ih);
	int nw = int(iw*scale);
	int nh = int(ih*scale);

	cv::Mat img_tmp;

	cv::resize(imgIn, img_tmp, cv::Size(nw, nh), CV_INTER_AREA);
	cv::Mat new_image = cv::Mat::Mat(cv::Size(input_height, input_width), CV_8UC3, cv::Scalar(128, 128, 128));
	cv::Mat new_imageROI = new_image(cv::Rect(int((input_width - nw) / 2), int((input_height - nh) / 2), nw, nh));
	img_tmp.copyTo(new_imageROI);
	*imgOut = new_image;
}

double calc_iou(mat box1, mat box2)
{
	double iou_result;
	float inter;
	float tb = std::min(box1(0, 2), box2(0, 2)) - std::max(box1(0, 0), box2(0, 0));
	float lr = std::min(box1(0, 3), box2(0, 3)) - std::max(box1(0, 1), box2(0, 1));
	if (tb < 0 || lr < 0)
		inter = 0;
	else
		inter = tb * lr;
	iou_result = inter / ((box1(0, 2) - box1(0, 0)) * (box1(0, 3) - box1(0, 1))
		+ (box2(0, 2) - box2(0, 0)) * (box2(0, 3) - box2(0, 1)) - inter);
	return iou_result;
}

void yolo_correct_boxes(arma::mat &output_dim3_mat,
	mat anchor_mask,
	int num_classes,
	std::vector<float> image_shape,
	mat offset,
	mat scale,
	float score_threshold,
	std::vector<mat> &boxes_scores)
{
	for (int i = 0; i < output_dim3_mat.n_cols; i++)
	{
		mat tmp_mat(1, 6);
		tmp_mat.zeros();

		output_dim3_mat(0, i) = (output_dim3_mat(0, i) - offset(0, 1)) * scale(0, 1);
		output_dim3_mat(1, i) = (output_dim3_mat(1, i) - offset(0, 0)) * scale(0, 0);
		output_dim3_mat(2, i) *= scale(0, 1);
		output_dim3_mat(3, i) *= scale(0, 0);

		tmp_mat(0, 1) = (output_dim3_mat(1, i) - (output_dim3_mat(3, i) / 2.0)) * image_shape[0];
		if (tmp_mat(0, 1) < 0)
			tmp_mat(0, 1) = 0;
		tmp_mat(0, 0) = (output_dim3_mat(0, i) - (output_dim3_mat(2, i) / 2.0)) * image_shape[1];
		if (tmp_mat(0, 0) < 0)
			tmp_mat(0, 0) = 0;
		tmp_mat(0, 3) = (output_dim3_mat(1, i) + (output_dim3_mat(3, i) / 2.0)) * image_shape[0];
		if (tmp_mat(0, 3) > image_shape[0])
			tmp_mat(0, 3) = image_shape[0];
		tmp_mat(0, 2) = (output_dim3_mat(0, i) + (output_dim3_mat(2, i) / 2.0)) * image_shape[1];
		if (tmp_mat(0, 2) > image_shape[1])
			tmp_mat(0, 2) = image_shape[1];

		for (int j = 5; j < output_dim3_mat.n_rows; j++)
		{
			if (output_dim3_mat(j, i) > score_threshold)
			{
				tmp_mat(0, 4) = output_dim3_mat(j, i);
				tmp_mat(0, 5) = j - 5;
				boxes_scores.push_back(tmp_mat);
			}
		}
	}

}

void yolo_eval(std::vector<Tensor> yolo_outputs,
	mat anchors,
	int num_classes,
	std::vector<float> image_shape,
	mat anchor_mask,
	std::vector<mat> &boxes_scores,
	float score_threshold = 0.6,
	float iou_threshold = 0.5)
{
	std::vector<float> input_shape;
	input_shape.push_back((yolo_outputs[0].shape().dim_size(1)) * 32);
	input_shape.push_back((yolo_outputs[0].shape().dim_size(2)) * 32);

	std::vector<float> new_shape;
	float m = std::min(input_shape[0] / image_shape[0], input_shape[1] / image_shape[1]);
	new_shape.push_back(round(m*image_shape[0]));
	new_shape.push_back(round(m*image_shape[1]));

	mat offset;
	offset << ((input_shape[0] - new_shape[0]) / (2.0*input_shape[0]))
		<< ((input_shape[1] - new_shape[1]) / (2.0*input_shape[1])) << endr;
	mat scale;
	scale << input_shape[0] / new_shape[0] << input_shape[1] / new_shape[1] << endr;
	std::vector<float>().swap(new_shape);

	for (int l = 0; l < yolo_outputs.size(); l++)
	{
		mat an_mask = anchor_mask.row(l);

		for (int i = 0; i < yolo_outputs[l].shape().dim_size(1); i++)
		{
			for (int j = 0; j < yolo_outputs[l].shape().dim_size(2); j++)
			{
				arma::mat output_dim3_mat(1, yolo_outputs[l].shape().dim_size(3));
				for (int k = 0; k < yolo_outputs[l].shape().dim_size(3); k++)
				{
					output_dim3_mat(0, k) = yolo_outputs[l].tensor<float, 4>()(0, i, j, k);
				}
				output_dim3_mat.reshape(num_classes + 5, an_mask.n_cols);
				for (int opt_dim3_mat_col = 0; opt_dim3_mat_col < output_dim3_mat.n_cols; opt_dim3_mat_col++)
				{
					for (int mat_row = 0; mat_row < output_dim3_mat.n_rows; mat_row++)
					{
						if (mat_row == 0)
						{
							output_dim3_mat(mat_row, opt_dim3_mat_col) = (sigmoid(output_dim3_mat(mat_row, opt_dim3_mat_col)) + j)
								/ (yolo_outputs[l].shape().dim_size(2));
						}
						else if (mat_row == 1)
						{
							output_dim3_mat(mat_row, opt_dim3_mat_col) = (sigmoid(output_dim3_mat(mat_row, opt_dim3_mat_col)) + i)
								/ (yolo_outputs[l].shape().dim_size(1));
						}
						else if (mat_row == 2)
						{
							output_dim3_mat(mat_row, opt_dim3_mat_col) = (exp(output_dim3_mat(mat_row, opt_dim3_mat_col))
								* anchors(int(an_mask(opt_dim3_mat_col)), 0)) / (input_shape[0]);
						}
						else if (mat_row == 3)
						{
							output_dim3_mat(mat_row, opt_dim3_mat_col) = (exp(output_dim3_mat(mat_row, opt_dim3_mat_col))
								* anchors(int(an_mask(opt_dim3_mat_col)), 1)) / (input_shape[1]);
						}
						else if (mat_row == 4)
						{
							output_dim3_mat(mat_row, opt_dim3_mat_col) = sigmoid(output_dim3_mat(mat_row, opt_dim3_mat_col));
						}
						else
						{
							output_dim3_mat(mat_row, opt_dim3_mat_col) = (sigmoid(output_dim3_mat(mat_row, opt_dim3_mat_col)))
								* output_dim3_mat(4, opt_dim3_mat_col);
						}
					}
				}
				yolo_correct_boxes(output_dim3_mat, an_mask, num_classes, image_shape, offset, scale, score_threshold, boxes_scores);
			}
		}
	}
}


float score;
float iou;
int image_size[2] = { 416, 416 };
std::string input_node0;
std::string output_node0;
std::string output_node1;
std::string output_node2;
int output_size;
std::vector<int> model_image_size(image_size, image_size + 2);
//std::vector<std::string> class_names;
int n_class;
//int anchors[6][2] = { { 10, 14 },{ 23, 27 },{ 37, 58 },{ 81, 82 },{ 135, 169 },{ 344, 319 } };
//int anchors[9][2] = { { 10, 13 },{ 16, 30 },{ 33, 23 },{ 30, 61 },{ 62, 45 },{ 59, 119 },{ 116, 90 },{ 156, 198 },{ 373, 326 } };
Session* session;
GraphDef graphdef;


int yolo_config(std::string model_path, 
	int classes_number, 
	int model_type, 
	float score_ = 0.2, 
	float iou_ = 0.45)
{
	//get_class(classes_path, &class_names);
	n_class = classes_number;
	switch (model_type)
	{
	case 0 :
		input_node0 = "input_1:0";
		output_node0 = "conv2d_10/BiasAdd:0";
		output_node1 = "conv2d_13/BiasAdd:0";
		output_size = 2;
		break;
	case 1:
		input_node0 = "input_1:0";
		output_node0 = "conv2d_16/BiasAdd:0";
		output_node1 = "conv2d_19/BiasAdd:0";
		output_size = 2;
		break;
	case 2:
		input_node0 = "input_1:0";
		output_node0 = "conv2d_12/BiasAdd:0";
		output_node1 = "conv2d_16/BiasAdd:0";
		output_node2 = "conv2d_20/BiasAdd:0";
		output_size = 3;
		break;
	default:
		std::cout << "input model type: " << model_type << std::endl;
		std::cout << "Model type error" << std::endl;
		return 0;
	}

	score = score_;
	iou = iou_;

	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok())
	{
		std::cout << status.ToString() << "\n";
		return 0;
	}
	std::cout << "Session created successfully" << std::endl;
	Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);
	if (!status_load.ok())
	{
		std::cout << "ERROR Loading model failed..." << model_path << std::endl;
		std::cout << status_load.ToString() << "\n";
		return 0;
	}
	std::cout << "Loading successfully" << std::endl;
	Status status_creat = session->Create(graphdef);
	if (!status_creat.ok())
	{
		std::cout << "ERROR: Creating graph in session failed..." << status_creat.ToString() << std::endl;
		return 0;
	}
	std::cout << "Created session and load graph successfully" << std::endl;
	return 1;
}

int yolo_img_detect(int height, int width, unsigned char *imageB, std::vector<std::vector<float>> &boxes)
{
	cv::Mat image(height, width, CV_8UC3, imageB);
	std::vector<float> input_image_shape;
	input_image_shape.push_back(image.rows);
	input_image_shape.push_back(image.cols);

	cv::Mat image_data;
	letterbox_image(image, model_image_size[0], model_image_size[1], &image_data);
	Tensor input_tensor(DT_FLOAT, TensorShape({ 1, model_image_size[0], model_image_size[1], 3 }));
	CVMat_to_Tensor(image_data, &input_tensor, model_image_size[0], model_image_size[1]);

	std::vector<Tensor> outputs;
	Status status_run;
	if (output_size == 2)
	{
		status_run = session->Run({ { input_node0, input_tensor } }, { output_node0, output_node1 }, {}, &outputs);
	}
	if (output_size == 3)
	{
		status_run = session->Run({ { input_node0, input_tensor } }, { output_node0, output_node1, output_node2 }, {}, &outputs);
	}
	if (!status_run.ok())
	{
		std::cout << "ERROR: RUN failed..." << std::endl;
		std::cout << status_run.ToString() << "\n";
		return 0;
	}

	std::vector<mat> boxes_scores;

	mat anchor_mask = get_anchors_mask(output_size);
	mat anchors = get_anchors(output_size);
	yolo_eval(outputs, anchors, n_class, input_image_shape, anchor_mask, boxes_scores, score, iou);
	if (boxes_scores.size() == 0)
	{
		std::vector<std::vector<float>>().swap(boxes);
		std::vector<mat>().swap(boxes_scores);
		std::vector<Tensor>().swap(outputs);
		std::vector<float>().swap(input_image_shape);
		return 1;
	}

	for (int i = 0; i < boxes_scores.size() - 1; i++)
	{
		if (boxes_scores[i](0, 4) > score)
		{
			for (int j = i + 1; j < boxes_scores.size(); j++)
			{
				if (boxes_scores[j](0, 4) > score)
				{
					//if (boxes_scores[i](0, 5) == boxes_scores[j](0, 5))
					//{
						if (calc_iou(boxes_scores[i], boxes_scores[j]) > iou)
						{
							if (boxes_scores[i](0, 4) > boxes_scores[j](0, 4))
								boxes_scores[j](0, 4) = 0;
							else
								boxes_scores[i](0, 4) = 0;
						}
					//}
				}
			}
		}
	}

	for (int i = 0; i < boxes_scores.size(); i++)
	{
		if (boxes_scores[i](0, 4) != 0)
		{
			std::vector<float> tmp;
			for (int j = 0; j < boxes_scores[i].size(); j++)
			{
				tmp.push_back(boxes_scores[i](0, j));
			}
			boxes.push_back(tmp);
			std::vector<float>().swap(tmp);
		}
	}

	std::vector<mat>().swap(boxes_scores);
	std::vector<Tensor>().swap(outputs);
	std::vector<float>().swap(input_image_shape);
	return 1;
}


