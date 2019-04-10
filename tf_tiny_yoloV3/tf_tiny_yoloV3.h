// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 TF_TINY_YOLOV3_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// TF_TINY_YOLOV3_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
//#pragma once

//#define COMPILER_MSVC
//#define NOMINMAX

//#ifdef TF_TINY_YOLOV3_EXPORTS
//#define TF_TINY_YOLOV3_API __declspec(dllexport)
//#else
//#define TF_TINY_YOLOV3_API __declspec(dllimport)
//#endif

#include <vector>

#ifdef TF_TINY_YOLOV3_EXPORTS
#define API extern  __declspec(dllexport) 
#else
#define API extern  __declspec(dllimport) 
#endif


/******************
model_path：pb文件路径
classes_number：检测目标的类别数量
full_conv_type：模型类别 0：tiny_yoloV3; 1: tiny_yoloV3_full_conv; 2: tiny_yolo_body_9an;
score_: 类别概率阈值
iou_：目标框重合度阈值
******************/
API int yolo_config(std::string model_path, int classes_number, int model_type, float score_, float iou_);

/******************
wigth，height: 图像的宽和高
imageB: 图像的buffer
boxes: [[lu_x, lu_y, rd_x, rd_y, probability, class],]
       lu_x,lu_y为左上角x,y坐标
	   rd_x,rd_y为右下角x,y坐标
	   probability为坐标区域目标类的概率
	   class为坐标区域的类别
******************/
API int yolo_img_detect(int height, int wigth, unsigned char *imageB, std::vector<std::vector<float>> &boxes);


