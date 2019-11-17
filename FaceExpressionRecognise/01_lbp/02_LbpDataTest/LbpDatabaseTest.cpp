//#pragma comment(linker, "/subsystem:windows /entry:mainCRTStartup")//去除CMD窗口

#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  
#include <io.h> //查找文件相关函数
#include "LBP.h"
#include "SvmTest.h"

#define NORM_WIDTH 140
#define NORM_HEIGHT 140

//
//#define 每种表情样本数目 100
#define 表情种类 7
//#define 特征向量维数 7*7*58

using namespace dlib;
using namespace std;
//using namespace cv;
using namespace cv::ml;

char * filePath[7] = {  "D:\\我的数据库\\JAFFE数据库\\jaffe0\\test\\1angry",   //图片数据集的目录
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\test\\2disgust",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\test\\3fear",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\test\\4happy",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\test\\5sadness",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\test\\6surprise",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\test\\7neutral"
						}; //样本路径
//char * filePath[7] = {  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\2disgust",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\3fear",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\4happy",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\5sadness",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\6surprise",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\7neutral"
//						}; //样本路径
//char * filePath[7] = {  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\test\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\test\\2disgust",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\test\\3fear",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\test\\4happy",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\test\\5sadness",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\test\\6surprise",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe1\\test\\7neutral"
//						}; //样本路径
//char * filePath[7] = {  "D:\\我的数据库\\JAFFE数据库\\jaffe0\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\2disgust",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\3fear",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\4happy",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\5sadness",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\6surprise",
//						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\7neutral"
						//}; //样本路径
//char * filePath[7] = { "D:\\我的数据库\\CK+new\\ck1\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\CK+new\\ck1\\2disgust",
//						"D:\\我的数据库\\CK+new\\ck1\\3fear",
//						"D:\\我的数据库\\CK+new\\ck1\\4happy",
//						"D:\\我的数据库\\CK+new\\ck1\\5sadness",
//						"D:\\我的数据库\\CK+new\\ck1\\6surprise",
//						"D:\\我的数据库\\CK+new\\ck1\\7neutral"
//						}; //样本路径
//char * filePath[7] = { "D:\\我的数据库\\CK+new\\CK+与人有关\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\2disgust",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\3fear",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\4happy",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\5sadness",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\6surprise",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\7neutral"
//						}; //样本路径
//char * filePath[7] = {  "D:\\我的数据库\\CK+new\\CK+与人有关\\test\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\test\\2disgust",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\test\\3fear",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\test\\4happy",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\test\\5sadness",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\test\\6surprise",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\test\\7neutral"
//						}; //样本路径
//
//char * filePath[7] = {  "D:\\我的数据库\\CK+new\\CK+与人无关\\test\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\test\\2disgust",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\test\\3fear",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\test\\4happy",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\test\\5sadness",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\test\\6surprise",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\test\\7neutral"
//						}; //样本路径

//char * filePath[7] = {	"D:\\我的数据库\\CK+new\\CK+与人无关\\1angry",   //图片数据集的目录
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\2disgust",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\3fear",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\4happy",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\5sadness",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\6surprise",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\7neutral"
//							}; //样本路径


string 表情名称[7] = { "1angry","2disgust","3fear","4happy","5sadness","6surprise","7neutral" };  //采集的文件的文件夹
string svmModelFilePath = "D:\\我的数据库\\JAFFE数据库\\jaffe0\\lbp_uniform_49\\Classifier.xml";
//string svmModelFilePath = "D:\\我的数据库\\CK+new\\CK+与人有关\\lbp_uniform_49\\Classifier.xml";

void getFiles(string path, std::vector<string>& files);//文件遍历

int main()
{

		double init = 0, face_t = 0, faceAll = 0, shape_t = 0, shapeAll = 0;//建立时间
		double all_time = 0, per_time = 0;
		int all_number = 0;
		float 最终结果[7] = { 0,0,0,0,0,0,0 };
		std::cout << "开始测试" << endl;
		int result;

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;

		init = (double)cv::getTickCount();///////////////////////////////////////模型加载时间
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		init = (double)cv::getTickCount() - init;//测试完毕

		cv::Ptr<SVM> svm = StatModel::load<SVM>(svmModelFilePath);//加载xml文件
		for (int j = 0; j < 7; j++)//分别将采集的数据存入7个文件夹中
		{
			int 表情[7] = { 0,0,0,0,0,0,0 };
			std::cout << "开始测试第" << j + 1 << "个文件" << endl;
			std::vector<string> files;//文件变量
			getFiles(filePath[j], files);		//文件遍历
			int number = files.size();			//获取文件夹中测试图片的数量
			all_number = all_number + number;
			for (int i = 0; i < number; i++)	//采集数据开始
			{

				cv::Mat temp = cv::imread(files[i].c_str());
				per_time = (double)cv::getTickCount();/////////////////////////////////////////人脸检测开始时间
				face_t = (double)cv::getTickCount();/////////////////////////////////////////人脸检测开始时间

				cv_image<bgr_pixel> cimg(temp);
				std::vector<dlib::rectangle> faces = detector(cimg);

				face_t = (double)cv::getTickCount() - face_t;////////////////////////////////人脸检测结束时间
				std::printf("face detection time = %g ms\n", face_t * 1000 / cv::getTickFrequency());
				faceAll = faceAll + face_t;
				face_t = 0;

				shape_t = (double)cv::getTickCount();////////////////////////////////////////人脸定位开始时间
				std::vector<full_object_detection> shapes;
				if (faces.empty()) {
				}
				else {
					for (unsigned long i = 0; i < faces.size(); ++i)
						shapes.push_back(pose_model(cimg, faces[i]));
				}
				double eyecenter_x1;
				double eyecenter_y1;
				double eyecenter_x2;
				double eyecenter_y2;
				double eyecenter_x;
				double eyecenter_y;
				double pointdistance;
				if (!shapes.empty()) {

					shape_t = (double)cv::getTickCount() - shape_t;
					std::printf("face loactaion time = %g ms\n", shape_t * 1000 / cv::getTickFrequency());//人脸定位结束时间
					shapeAll = shapeAll + shape_t;//人脸定位总时间
					shape_t = 0;

					eyecenter_x1 = (shapes[0].part(36).x() + shapes[0].part(37).x() + shapes[0].part(38).x() + shapes[0].part(39).x() + shapes[0].part(40).x() + shapes[0].part(41).x()) / 6.0;
					//	cout << "eyecenter_x1  " << eyecenter_x1 << endl;
					eyecenter_y1 = (shapes[0].part(36).y() + shapes[0].part(37).y() + shapes[0].part(38).y() + shapes[0].part(39).y() + shapes[0].part(40).y() + shapes[0].part(41).y()) / 6.0;
					//	cout << "eyecenter_y1  " << eyecenter_y1 << endl;
					eyecenter_x2 = (shapes[0].part(42).x() + shapes[0].part(43).x() + shapes[0].part(44).x() + shapes[0].part(45).x() + shapes[0].part(46).x() + shapes[0].part(47).x()) / 6.0;
					//	cout << "eyecenter_x2  " << eyecenter_x2 << endl;
					eyecenter_y2 = (shapes[0].part(42).y() + shapes[0].part(43).y() + shapes[0].part(44).y() + shapes[0].part(45).y() + shapes[0].part(46).y() + shapes[0].part(47).y()) / 6.0;
					eyecenter_x = (eyecenter_x1 + eyecenter_x2) / 2;
					eyecenter_y = (eyecenter_y1 + eyecenter_y2) / 2;
					pointdistance = sqrt(pow(eyecenter_x1 - eyecenter_x2, 2) + pow(eyecenter_y1 - eyecenter_y2, 2));  //计算均方差
				}
				//从数据集中读取图片，并转化为灰度图像
				IplImage* srcImg_gary = cvLoadImage(files[i].c_str(), 0);//从数据库中得到的原始图像1，0灰色、1彩色
				//将灰度图像进行直方图均衡
				IplImage* srcImg_equalize = cvCreateImage(cvGetSize(srcImg_gary), IPL_DEPTH_8U, 1);
				cvEqualizeHist(srcImg_gary, srcImg_equalize);////////////////////////直方图均衡
				 //将均衡化后的图像进行裁剪，得到裁剪之后的图像
				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//选择区域大小
				cvSetImageROI(srcImg_equalize, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//设置源图像ROI
				IplImage* srcImg_ori = cvCreateImage(size, srcImg_equalize->depth, srcImg_equalize->nChannels);//创建空白的目标图像
				cvCopy(srcImg_equalize, srcImg_ori); //复制图像：srcImg->pDest
				//将裁剪之后的图像进行尺度归一化
				IplImage* srcImg_resize;
				CvSize norm_cvsize;
				norm_cvsize.width = NORM_WIDTH;  //目标图像的宽    
				norm_cvsize.height = NORM_HEIGHT; //目标图像的高  
				srcImg_resize = cvCreateImage(norm_cvsize, srcImg_ori->depth, srcImg_ori->nChannels);//构造目标图象  
				cvResize(srcImg_ori, srcImg_resize, CV_INTER_LINEAR); //缩放源图像到目标图像 
				//cvNamedWindow("srcImg_resize", CV_WINDOW_AUTOSIZE);
				//cvShowImage("srcImg_resize", srcImg_resize);//归一化的图像
				//cvWaitKey(0);
				//cv::Mat srcHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
				//cv::Mat dstHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
				//drawHistImg(cv::cvarrToMat(srcImg_gary), srcHistImage, "srcHistImage");
				//drawHistImg(cv::cvarrToMat(srcImg_equalize), dstHistImage, "dstHistImage");
				//cvWaitKey(0);
				printf("lbp get feature...\n");
				Mat featureVectorOfTestImage;//新建特征向量
				LBP lbp;
				lbp.ComputeLBPFeatureVector_Uniform(cv::cvarrToMat(srcImg_resize), Size(CELL_SIZE, CELL_SIZE), featureVectorOfTestImage);
				if (featureVectorOfTestImage.empty())
					continue;
				int predictResult=svm->predict(featureVectorOfTestImage * 1000);//开始预测

				switch (predictResult)
				{
						case 200:表情[0]++; break;
						case 250:表情[1]++; break;
						case 300:表情[2]++; break;
						case 350:表情[3]++; break;
						case 400:表情[4]++; break;
						case 450:表情[5]++; break;
						case 500:表情[6]++; break;
				}
				per_time = (double)cv::getTickCount() - per_time;////////////////////////////////人脸识别结束时间
				all_time = all_time + per_time;
				printf("recognition time = %g ms per image\n", per_time * 1000 / cv::getTickFrequency());//////////////////////测量结束
				per_time = 0;

				std::cout <<  "完成率:" << float(i+1)*100/number << "%" << endl;

			}//每一种表情的文件夹文件目录下for循环
			std::cout << 表情名称[j] << "中含有:" << number << "张测试图片" << endl;
			for (int x = 0; x < 7; x++)
			{
				std::cout << 表情名称[x] << "：" << 表情[x] << endl;
			}
			std::cout << "识别率：" << (float)表情[j] / number << endl;
			最终结果[j] = (float)表情[j] / number;

		}//7种表情的for
		for (int y = 0; y < 7; y++)
		{
			std::cout << 表情名称[y] << "的识别率：" << 最终结果[y] << endl;
		}
		std::cout << "平均识别率：" << float(最终结果[0] + 最终结果[1] + 最终结果[2] + 最终结果[3] + 最终结果[4] + 最终结果[5] + 最终结果[6]) / 7 << endl;
		std::cout << "加载模型的总时间:" << init * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "人脸检测总时间:  " << faceAll * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "人脸检测平均时间:" << faceAll * 1000 / cv::getTickFrequency() / all_number << "ms" << endl;
		std::cout << "人脸定位的总时间:" << shapeAll * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "人脸定位平均时间:" << shapeAll * 1000 / cv::getTickFrequency() / all_number << "ms" << endl;
		std::cout << "表情识别总时间:  " << all_time * 1000 / cv::getTickFrequency()  << "ms" << endl;
		std::cout << "表情识别平均时间:" << all_time * 1000 / cv::getTickFrequency() / all_number << "ms" << endl;
		
		std::system("pause");



}
void getFiles(string path, std::vector<string>& files)	//文件遍历函数
{
	intptr_t   hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	int i = 30;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}

		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}
