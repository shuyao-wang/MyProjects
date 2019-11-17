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
#define 每种表情样本数目 24
#define 表情种类 7

using namespace dlib;
using namespace std;
//using namespace cv;
using namespace cv::ml;

//
//char * filePath[7] = {  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\1angry",   //图片数据集的目录
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\2disgust",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\3fear",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\4happy",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\5sadness",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\6surprise",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\7neutral"
//						}; //样本路径

char * filePath[7] = {  "D:\\我的数据库\\JAFFE数据库\\jaffe0\\1angry",   //图片数据集的目录
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\2disgust",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\3fear",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\4happy",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\5sadness",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\6surprise",
						"D:\\我的数据库\\JAFFE数据库\\jaffe0\\7neutral"
}; //样本路径

//char * filePath[7] =  { "D:\\我的数据库\\CK+new\\CK+与人无关\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\2disgust",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\3fear",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\4happy",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\5sadness",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\6surprise",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\7neutral"
//						}; //样本路径
//char * filePath[7] = {	"D:\\我的数据库\\CK+new\\CK+与人无关\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\2disgust",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\3fear",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\4happy",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\5sadness",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\6surprise",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\7neutral"
//						}; //样本路径
//char * filePath[7] = {	"D:\\我的数据库\\CK+new\\CK+与人有关\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\2disgust",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\3fear",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\4happy",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\5sadness",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\6surprise",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\7neutral"
//						}; //样本路径

string 文件夹名称[7] = { "1angry","2disgust","3fear","4happy","5sadness","6surprise","7neutral" };  //采集的文件的文件夹
//string svmModelFilePath = "D:\\我的数据库\\CK+new\\CK+与人有关\\lbp_uniform_49\\Classifier.xml";
string svmModelFilePath = "D:\\我的数据库\\JAFFE数据库\\jaffe0\\lbp_uniform_49\\Classifier.xml";
void getFiles(string path, std::vector<string>& files);//文件遍历
void drawHistImg(cv::Mat &src, cv::Mat &histImage, std::string name);//直方图绘制
int main()
{

	double init = 0, face_t = 0, faceAll = 0, shape_t = 0, shapeAll = 0;//建立时间
	cout << "==================================现在开始采集数据====================================" << endl;
	cout << "                                                                                      " << endl;
	cout << "                                                                                      " << endl;
	Mat featureVectorsOfSample;	//建立特征向量矩阵
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;


	init = (double)cv::getTickCount();///////////////////////////////////////模型加载时间
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	init = (double)cv::getTickCount() - init;//测试完毕


	for (int j = 0; j < 7; j++)//分别将采集的数据存入7个文件夹中
	{
		cout << "%%%%%%%%%%%%%%%%%%%%%%%正在采集:" << 文件夹名称[j] << "表情数据%%%%%%%%%%%%%%%%%%%%%%%" << endl;
		std::vector<string> files;
		getFiles(filePath[j], files);		//文件遍历
		for (int i = 0; i < 每种表情样本数目; i++)	//采集数据开始
		{
			cv::Mat temp = cv::imread(files[i].c_str());

			face_t = (double)cv::getTickCount();/////////////////////////////////////////人脸检测开始时间

			cv_image<bgr_pixel> cimg(temp);
			std::vector<dlib::rectangle> faces = detector(cimg);

			face_t = (double)cv::getTickCount() - face_t;////////////////////////////////人脸检测结束时间
			std::printf("face detection time = %g ms\n", face_t * 1000 / cv::getTickFrequency());
			faceAll = faceAll + face_t;
			face_t = 0;

			std::vector<full_object_detection> shapes;

			shape_t = (double)cv::getTickCount();////////////////////////////////////////人脸定位开始时间
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
			//cvNamedWindow("srcImg_gary", CV_WINDOW_AUTOSIZE);
			//cvShowImage("srcImg_gary", srcImg_gary);
			//将灰度图像进行直方图均衡
			IplImage* srcImg_equalize = cvCreateImage(cvGetSize(srcImg_gary), IPL_DEPTH_8U, 1);
			cvEqualizeHist(srcImg_gary, srcImg_equalize);////////////////////////直方图均衡
			//cvNamedWindow("srcImg_equalize", CV_WINDOW_AUTOSIZE);
			//cvShowImage("srcImg_equalize", srcImg_equalize);//归一化的图像
			//将均衡化后的图像进行裁剪，得到裁剪之后的图像
			CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//选择区域大小
			cvSetImageROI(srcImg_equalize, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//设置源图像ROI
			IplImage* srcImg_ori = cvCreateImage(size, srcImg_equalize->depth, srcImg_equalize->nChannels);//创建空白的目标图像
			cvCopy(srcImg_equalize, srcImg_ori); //复制图像：srcImg->pDest
			//cvNamedWindow("srcImg_ori", CV_WINDOW_AUTOSIZE);
			//cvShowImage("srcImg_ori", srcImg_ori);//最原始的RGB图像
			//cvWaitKey(0);
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
			////绘制灰度直方图
			//cv::Mat srcHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
			//cv::Mat dstHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
			//drawHistImg(cv::cvarrToMat(srcImg_gary), srcHistImage, "srcHistImage");
			//drawHistImg(cv::cvarrToMat(srcImg_equalize), dstHistImage, "dstHistImage");
			//cvWaitKey(0);
			printf("lbp get feature...\n");
			Mat featureVector;//每个样本的特征向量
			LBP lbp;
			lbp.ComputeLBPFeatureVector_Uniform(cv::cvarrToMat(srcImg_resize), Size(CELL_SIZE, CELL_SIZE), featureVector);
			if (featureVector.empty())
				continue;
			featureVectorsOfSample.push_back(featureVector);

			cout << "完成第" << i + 1 << "个LBP特征向量提取" << endl;
			cout << "第" << j + 1 << "个目录完成率："<<(float(i+1)/ 每种表情样本数目)*100<<"%" << endl;
		}
	}
			cout << "加载模型的总时间:" << init * 1000 / cv::getTickFrequency() <<"ms"<< endl;
			cout << "人脸检测总时间:  " << faceAll * 1000 / cv::getTickFrequency() << "ms" << endl;
			cout << "人脸定位的总时间:" << shapeAll * 1000 / cv::getTickFrequency() << "ms" << endl;
			cout << "人脸检测平均时间:" << faceAll * 1000 / cv::getTickFrequency() / (7 * 每种表情样本数目) << "ms" << endl;
			cout << "人脸定位平均时间:" << shapeAll * 1000 / cv::getTickFrequency() / (7 * 每种表情样本数目) << "ms" << endl;
	// train
	printf("training...\n");
	double time1, time2;

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setC(1);
	svm->setCoef0(0);
	svm->setDegree(0);
	svm->setGamma(1);
	svm->setNu(0);
	svm->setP(0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	int 人脸标签[每种表情样本数目*表情种类];	//203个样本标签
	for (int j = 0; j < 表情种类; j++) {

		for (int i = 0; i < 每种表情样本数目; i++) {   //为标签赋值
			人脸标签[i + 每种表情样本数目*j] = 200 + 50 * j;     //标签为200、250、300、350、400、450、500
		}
	}
	Mat 标签label(每种表情样本数目*表情种类, 1, CV_32SC1, 人脸标签);//标签数据转换
	time1 = getTickCount();
	svm->train(featureVectorsOfSample * 1000, ROW_SAMPLE, 标签label);//SVM训练，featureVectorsOfSample为输入的特征向量，classOfSample为标签
	time2 = getTickCount();
	printf("训练时间:%fms\n", (time2 - time1)*1000. / getTickFrequency());
	printf("training done!\n");
	svm->save(svmModelFilePath);//模型存储
	system("pause");
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

void drawHistImg(cv::Mat &src, cv::Mat &histImage, std::string name)
{
	const int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	cv::MatND hist;
	int channels[] = { 0 };
	cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);
	double maxValue;
	cv::minMaxLoc(hist, 0, &maxValue, 0, 0);
	int scale = 1;
	int histHeight = 256;
	for (int i = 0; i < bins; i++)
	{
		float binValue = hist.at<float>(i);
		int height = cvRound(binValue*histHeight / maxValue);
		cv::rectangle(histImage, cv::Point(i*scale, histHeight), cv::Point((i + 1)*scale, histHeight - height), cv::Scalar(255));

		cv::imshow(name, histImage);
	}
}

////#pragma comment(linker, "/subsystem:windows /entry:mainCRTStartup")//去除CMD窗口
//
//#include <dlib/opencv.h>  
//#include <opencv2/opencv.hpp>  
//#include <dlib/image_processing/frontal_face_detector.h>  
//#include <dlib/image_processing/render_face_detections.h>  
//#include <dlib/image_processing.h>  
//#include <dlib/gui_widgets.h>  
//#include <io.h> //查找文件相关函数
//#include "LBP.h"
//#include "SvmTest.h"
//
//#define NORM_WIDTH 140
//#define NORM_HEIGHT 140
//#define 每种表情样本数目 90
//#define 表情种类 7
////#define 特征向量维数 7*7*58+4*4*58
//
//using namespace dlib;
//using namespace std;
////using namespace cv;
//using namespace cv::ml;
//
////
////char * filePath[7] = {    "D:\\我的数据库\\JAFFE数据库\\jaffe1\\6surprise",   //图片数据集的目录
////						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\2disgust",
////						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\3fear",
////						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\4happy",
////						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\5sadness",
////						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\6surprise",
////						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\7neutral"
////						}; //样本路径
//
////char * filePath[7] =  { "D:\\我的数据库\\CK+new\\CK+与人无关\\1angry",   //图片数据集的目录
////						"D:\\我的数据库\\CK+new\\CK+与人无关\\2disgust",
////						"D:\\我的数据库\\CK+new\\CK+与人无关\\3fear",
////						"D:\\我的数据库\\CK+new\\CK+与人无关\\4happy",
////						"D:\\我的数据库\\CK+new\\CK+与人无关\\5sadness",
////						"D:\\我的数据库\\CK+new\\CK+与人无关\\6surprise",
////						"D:\\我的数据库\\CK+new\\CK+与人无关\\7neutral"
////						}; //样本路径
//char * filePath[7] = { "D:\\我的数据库\\CK+new\\CK+与人无关\\1angry1",   //图片数据集的目录
//"D:\\我的数据库\\CK+new\\CK+与人无关\\2disgust",
//"D:\\我的数据库\\CK+new\\CK+与人无关\\3fear",
//"D:\\我的数据库\\CK+new\\CK+与人无关\\4happy",
//"D:\\我的数据库\\CK+new\\CK+与人无关\\5sadness",
//"D:\\我的数据库\\CK+new\\CK+与人无关\\6surprise",
//"D:\\我的数据库\\CK+new\\CK+与人无关\\7neutral"
//}; //样本路径
//
//string 文件夹名称[7] = { "1angry","2disgust","3fear","4happy","5sadness","6surprise","7neutral" };  //采集的文件的文件夹
////string svmModelFilePath = "D:\\我的数据库\\JAFFE数据库\\jaffe0\\lbp_uniform_492\\Classifier.xml";
////string svmModelFilePath = "D:\\我的数据库\\JAFFE数据库\\jaffe1\\lbp_uniform_64\\Classifier.xml";
////string svmModelFilePath = "D:\\我的数据库\\A_我的表情数据集\\my_emotion\\Classifier.xml";
//string svmModelFilePath = "D:\\我的数据库\\CK+new\\CK+与人无关\\lbp_uniform_49\\Classifier.xml";
//void getFiles(string path, std::vector<string>& files);//文件遍历
//void drawHistImg(cv::Mat &src, cv::Mat &histImage, std::string name);//直方图绘制
//int main()
//{
//		double t = 0;
//		cout << "==================================现在开始采集数据====================================" << endl;
//		cout << "                                                                                      " << endl;
//		cout << "                                                                                      " << endl;
//		Mat featureVectorsOfSample;	//建立特征向量矩阵
//		frontal_face_detector detector = get_frontal_face_detector();
//		shape_predictor pose_model;
//		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
//		std::vector<dlib::rectangle> faces2;
//		for (int j = 0; j < 7; j++)//分别将采集的数据存入7个文件夹中
//		{
//			cout << "%%%%%%%%%%%%%%%%%%%%%%%正在采集:" << 文件夹名称[j] << "表情数据%%%%%%%%%%%%%%%%%%%%%%%" << endl;
//			std::vector<string> files;
//			getFiles(filePath[j], files);		//文件遍历
//			for (int i = 0; i < 每种表情样本数目; i++)	//采集数据开始
//			{
//				t = (double)cv::getTickCount();///////////////////////////////////////开始检测时间
//				cv::Mat temp = cv::imread(files[i].c_str());
//				cv_image<bgr_pixel> cimg(temp);
//				std::vector<dlib::rectangle> faces = detector(cimg);
//				if (!faces.empty())faces2 = faces;
//				//t = (double)cv::getTickCount() - t;
//				//printf("detection time = %g ms\n", t * 1000 / cv::getTickFrequency());//////////////////////测量结束
//				std::vector<full_object_detection> shapes;
//				if (faces.empty()) {
//				}
//				else {
//					for (unsigned long i = 0; i < faces.size(); ++i)
//						shapes.push_back(pose_model(cimg, faces[i]));
//				}
//				double eyecenter_x1;
//				double eyecenter_y1;
//				double eyecenter_x2;
//				double eyecenter_y2;
//				double eyecenter_x;
//				double eyecenter_y;
//				double pointdistance;
//				if (!shapes.empty()) {
//					eyecenter_x1 = (shapes[0].part(36).x() + shapes[0].part(37).x() + shapes[0].part(38).x() + shapes[0].part(39).x() + shapes[0].part(40).x() + shapes[0].part(41).x()) / 6.0;
//					//	cout << "eyecenter_x1  " << eyecenter_x1 << endl;
//					eyecenter_y1 = (shapes[0].part(36).y() + shapes[0].part(37).y() + shapes[0].part(38).y() + shapes[0].part(39).y() + shapes[0].part(40).y() + shapes[0].part(41).y()) / 6.0;
//					//	cout << "eyecenter_y1  " << eyecenter_y1 << endl;
//					eyecenter_x2 = (shapes[0].part(42).x() + shapes[0].part(43).x() + shapes[0].part(44).x() + shapes[0].part(45).x() + shapes[0].part(46).x() + shapes[0].part(47).x()) / 6.0;
//					//	cout << "eyecenter_x2  " << eyecenter_x2 << endl;
//					eyecenter_y2 = (shapes[0].part(42).y() + shapes[0].part(43).y() + shapes[0].part(44).y() + shapes[0].part(45).y() + shapes[0].part(46).y() + shapes[0].part(47).y()) / 6.0;
//					eyecenter_x = (eyecenter_x1 + eyecenter_x2) / 2;
//					eyecenter_y = (eyecenter_y1 + eyecenter_y2) / 2;
//					pointdistance = sqrt(pow(eyecenter_x1 - eyecenter_x2, 2) + pow(eyecenter_y1 - eyecenter_y2, 2));  //计算均方差
//					for (int i = 0; i < 68; i++) {
//						circle(temp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
//					}
//				}
//				//cv::imshow("人脸关键点定位",temp);
//				//waitKey(0);
//				IplImage* srcImg = cvLoadImage(files[i].c_str(), 0);//从数据库中得到的原始图像1
//				//IplImage* srcImg = cvCreateImage(cvGetSize(srcImg1), IPL_DEPTH_8U, 1);
//				//cvEqualizeHist(srcImg1, srcImg);////////////////////////直方图均衡
//				//cvNamedWindow("Result1", CV_WINDOW_AUTOSIZE);
//				//cvShowImage("Result1", srcImg);
//				//CvSize size = cvSize(faces[0].right() - faces[0].left(), faces[0].bottom() - faces[0].top());//选择区域大小
//				//cvSetImageROI(srcImg, cvRect(faces[0].left(), faces[0].top(), size.width, size.height));//设置源图像ROI
//				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//选择区域大小
//				cvSetImageROI(srcImg, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//设置源图像ROI
//				IplImage* pDest = cvCreateImage(size, srcImg->depth, srcImg->nChannels);//创建空白的目标图像
//				cvCopy(srcImg, pDest); //复制图像：srcImg->pDest
//				cv::Mat srcImage_1 = cv::cvarrToMat(pDest);
//				//cvNamedWindow("Result2", CV_WINDOW_AUTOSIZE);
//				//cvShowImage("Result2", pDest);
//				Mat srcImage(NORM_WIDTH, NORM_HEIGHT, srcImage_1.type());
//				resize(srcImage_1, srcImage, srcImage.size(), 0, 0, INTER_LINEAR);//双线性插值算法
//
//
//				cvNamedWindow("灰度图", CV_WINDOW_AUTOSIZE);
//				cv::imshow("灰度图", srcImage);
//				cvWaitKey(0);
//
//				//IplImage* srcImg1= &IplImage(srcImage);
//				IplImage* srcImg1 = cvCreateImage(cvGetSize(&IplImage(srcImage)), IPL_DEPTH_8U, 1);
//				cvEqualizeHist(&IplImage(srcImage), srcImg1);////////////////////////直方图均衡
//				cvNamedWindow("直方图均衡", CV_WINDOW_AUTOSIZE);
//				cvShowImage("直方图均衡", srcImg1);
//				cvWaitKey(0);
//
//				cv::Mat srcHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
//				cv::Mat dstHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
//				drawHistImg(srcImage, srcHistImage, "srcHistImage");
//				drawHistImg(cv::cvarrToMat(srcImg1), dstHistImage, "dstHistImage");
//
//				cvWaitKey(0);
//
//				printf("get feature...\n");
//				LOG_INFO_SVM_TEST("get feature...");
//				Mat featureVector;//新建特征向量
//				LBP lbp;
//				//lbp.ComputeLBPFeatureVector_Rotation_Uniform(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVector);//提取特征向量，利用的旋转不变等价模式
//				lbp.ComputeLBPFeatureVector_Uniform(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVector);
//				//lbp.ComputeLBPFeatureVector_Uniform(srcImage, Size(42, 42), featureVector2);
//				//cv::hconcat(featureVector1, featureVector2, featureVector);
//				if (featureVector.empty())
//					continue;
//				featureVectorsOfSample.push_back(featureVector);
//				//featureVectorsOfSample.push_back(featureVector1);
//				cout << "完成打标签" << i + 1 << "个样本文件" << endl;
//				cout << "完成打标签" << j + 1 << "和数据的读入" << endl;
//				t = (double)cv::getTickCount() - t;
//				printf("detection time = %g ms\n", t * 1000 / cv::getTickFrequency());//////////////////////测量结束
//			}
//		}
//		// train
//		printf("training...\n");
//		LOG_INFO_SVM_TEST("training...");
//		double time1, time2;
//
//		Ptr<SVM> svm = SVM::create();
//		svm->setType(SVM::C_SVC);
//		svm->setKernel(SVM::LINEAR);
//		svm->setC(1);
//		svm->setCoef0(0);
//		svm->setDegree(0);
//		svm->setGamma(1);
//		svm->setNu(0);
//		svm->setP(0);
//		svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
//
//		int 人脸标签[每种表情样本数目*表情种类];	//203个样本标签
//		for (int j = 0; j < 表情种类; j++) {
//
//			for (int i = 0; i < 每种表情样本数目; i++) {   //为标签赋值
//				人脸标签[i + 每种表情样本数目*j] = 200 + 50 * j;     //标签为200、250、300、350、400、450、500
//			}
//		}
//		Mat 标签label(每种表情样本数目*表情种类, 1, CV_32SC1, 人脸标签);//标签数据转换
//		time1 = getTickCount();
//		svm->train(featureVectorsOfSample * 1000, ROW_SAMPLE, 标签label);//SVM训练，featureVectorsOfSample为输入的特征向量，classOfSample为标签
//		time2 = getTickCount();
//		printf("训练时间:%f\n", (time2 - time1)*1000. / getTickFrequency());
//		LOG_INFO_SVM_TEST("训练时间:%f", (time2 - time1)*1000. / getTickFrequency());
//		printf("training done!\n");
//		LOG_INFO_SVM_TEST("training done!");
//		// save model
//		svm->save(svmModelFilePath);//模型存储
//}
//
//void getFiles(string path, std::vector<string>& files)	//文件遍历函数
//{
//	intptr_t   hFile = 0;
//	struct _finddata_t fileinfo;
//	string p;
//	int i = 30;
//	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
//	{
//		do
//		{
//			if ((fileinfo.attrib &  _A_SUBDIR))
//			{
//				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
//					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
//			}
//			else
//			{
//				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
//			}
//
//		} while (_findnext(hFile, &fileinfo) == 0);
//
//		_findclose(hFile);
//	}
//}
//
//void drawHistImg(cv::Mat &src, cv::Mat &histImage, std::string name)
//{
//	const int bins = 256;
//	int hist_size[] = { bins };
//	float range[] = { 0, 256 };
//	const float* ranges[] = { range };
//	cv::MatND hist;
//	int channels[] = { 0 };
//
//	cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);
//
//	double maxValue;
//	cv::minMaxLoc(hist, 0, &maxValue, 0, 0);
//	int scale = 1;
//	int histHeight = 256;
//
//	for (int i = 0; i < bins; i++)
//	{
//		float binValue = hist.at<float>(i);
//		int height = cvRound(binValue*histHeight / maxValue);
//		cv::rectangle(histImage, cv::Point(i*scale, histHeight), cv::Point((i + 1)*scale, histHeight - height), cv::Scalar(255));
//
//		cv::imshow(name, histImage);
//	}
//}
//
