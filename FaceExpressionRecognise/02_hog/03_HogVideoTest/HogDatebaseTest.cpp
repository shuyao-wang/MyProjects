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

//#define NORM_WIDTH 126
//#define NORM_HEIGHT 126

#define capwidth 640
#define caphight 480

#define PI 3.14
#define BIN_SIZE 20//bin的20°划分
#define BIN_NVM 9//bin的数量9个
#define NORM_WIDTH 128//图片大小转换，宽度130
#define NORM_HEIGHT 128//图片大小转换，宽度82
//#define CELL_SIZE 8//cell为8*8大小
#define CELL_SIZE 14//cell为8*8大小
#define BLOCK_SIZE 2//block的大小2*2
//#define PIC_CELL_WH 50
#define CELL_W_NVM  ((NORM_WIDTH-2) / CELL_SIZE)//cell的横向数量
#define CELL_H_NVM  ((NORM_HEIGHT-2) / CELL_SIZE)//cell的纵向数量
#define BLOCK_W_NVM  (CELL_W_NVM - BLOCK_SIZE + 1)//block的横向数量16-1
#define BLOCK_H_NVM  (CELL_H_NVM - BLOCK_SIZE + 1)//block的纵向数量
#define CELL_NVM (CELL_W_NVM * CELL_H_NVM)//cell总数
#define BLOCK_NVM (BLOCK_W_NVM * BLOCK_H_NVM)//block总数
#define ARRAY_ALL (BLOCK_W_NVM * BLOCK_H_NVM * BLOCK_SIZE * BLOCK_SIZE * BIN_NVM)//block总数15*15*2*2*9，特征向量的维数

#define 扩大因子 1000

using namespace dlib;
using namespace std;
//using namespace cv;
using namespace cv::ml;
string svmModelFilePath = "D:\\我的数据库\\CK+new\\CK+与人无关\\hog_8×8\\Classifier.xml";
void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin);//输入的是cell左上角x,y，i_w是图片的宽度，Img_in是输入图片，fbin是输出的9维bin数组。

int main()
{

		double t = 0, t1 = 0;
		double fps;
		char string[10];  // 用于存放帧率的字符串
		cv::VideoCapture cap(0);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			system("pause");
		}
		cap.set(CV_CAP_PROP_FRAME_WIDTH, capwidth);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, caphight);
		//float f_Last_Array[ARRAY_ALL];//最终的输出向量

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		cv::Ptr<SVM> svm = StatModel::load<SVM>(svmModelFilePath);//加载xml文件

		while (cv::waitKey(30) != 27)
		{
			cv::Mat temp, temp1;
			cap >> temp;
			cap >> temp1;
			t = (double)cv::getTickCount();///////////////////////////////////////开始检测时间
			cv_image<bgr_pixel> cimg(temp);
			std::vector<dlib::rectangle> faces = detector(cimg);
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
			int predictResult = 0;
			if (!shapes.empty()) {
				eyecenter_x1 = (shapes[0].part(36).x() + shapes[0].part(37).x() + shapes[0].part(38).x() + shapes[0].part(39).x() + shapes[0].part(40).x() + shapes[0].part(41).x()) / 6.0;
				eyecenter_y1 = (shapes[0].part(36).y() + shapes[0].part(37).y() + shapes[0].part(38).y() + shapes[0].part(39).y() + shapes[0].part(40).y() + shapes[0].part(41).y()) / 6.0;
				eyecenter_x2 = (shapes[0].part(42).x() + shapes[0].part(43).x() + shapes[0].part(44).x() + shapes[0].part(45).x() + shapes[0].part(46).x() + shapes[0].part(47).x()) / 6.0;
				eyecenter_y2 = (shapes[0].part(42).y() + shapes[0].part(43).y() + shapes[0].part(44).y() + shapes[0].part(45).y() + shapes[0].part(46).y() + shapes[0].part(47).y()) / 6.0;
				eyecenter_x = (eyecenter_x1 + eyecenter_x2) / 2;
				eyecenter_y = (eyecenter_y1 + eyecenter_y2) / 2;
				pointdistance = sqrt(pow(eyecenter_x1 - eyecenter_x2, 2) + pow(eyecenter_y1 - eyecenter_y2, 2));  //计算均方差	
				for (int i = 0; i < 68; i++) {
					circle(temp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 2, cv::Scalar(255, 0, 0), -1);
					//  shapes[0].part(i).x();//68个  
				}
				cv::Mat dst;//
				cvtColor(temp1, dst, CV_BGR2GRAY);//将摄像头得到的temp1图像转换成单通道图像
				IplImage* srcImg = &IplImage(dst);//转化为灰度图
				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//选择区域大小
				if ((eyecenter_x - 0.9*pointdistance > 0) && (eyecenter_y - 0.5*pointdistance > 0) && (eyecenter_x + 0.9*pointdistance < capwidth) && (eyecenter_y + 2 * pointdistance < caphight))
				{
					cvSetImageROI(srcImg, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//设置源图像ROI
				}//判断是否超出范围
				else//超出范围则提示
				{
					cout << "超出范围！" << endl;
					continue;
				}
				IplImage* pDest = cvCreateImage(size, srcImg->depth, srcImg->nChannels);//创建空白的目标图像
				cvCopy(srcImg, pDest); //复制图像：srcImg->pDest
				cvResetImageROI(srcImg);//释放原ROI
				cv::Mat srcImage_1 = cv::cvarrToMat(pDest);
				Mat srcImage(NORM_HEIGHT, NORM_WIDTH, srcImage_1.type());
				resize(srcImage_1, srcImage, srcImage.size(), 0, 0, INTER_LINEAR);
				cv::imshow("Result1", srcImage);//预处理之后的输出

				float f_Last_Array[ARRAY_ALL];//最终的输出向量
				IplImage* img = &IplImage(srcImage);
				//IplImage *img1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
				IplImage *img2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
				CvMat* mat = cvCreateMat(img->width, img->height, CV_32FC1);
				//// gamma校正
				uchar* uData = (uchar*)(img->imageData);
				float* fMat = mat->data.fl;

				for (int ii = 0; ii < img->imageSize; ii++)
				{
					fMat[ii] = pow(uData[ii], 0.5f);
					((uchar*)img2->imageData)[ii] = (uchar)(fMat[ii]);
				}
				//// 计算每个cell每个梯度的大小和方向
				int i_binNvm = 0;
				float f_bin_out[CELL_NVM][BIN_NVM];//cell总数和bin的9个
				float i_AllbinNvm[][BLOCK_SIZE*BLOCK_SIZE*BIN_NVM] = { 0.0f };//2*2*9=36
				int ii_nvm1 = 0, ii_nvm2 = 0;
				for (int ii = 1; ii + CELL_SIZE < img2->height; ii += CELL_SIZE)
				{
					for (int jj = 1; jj + CELL_SIZE < img2->width; jj += CELL_SIZE)
					{
						func(jj, ii, CELL_SIZE, img2, f_bin_out[i_binNvm++]);//输出f_bin_out结果为所有的bin对应的9维数据
					}
				}
				int iBlockWhichCell = 0;
				int uu = 0;
				float  f_max = 0.0f;
				float f_Ether_Block[BLOCK_SIZE*BLOCK_SIZE][BIN_NVM];//每一个block中的4个cell
																	//float f_Last_Array[每种表情样本数目*表情种类][ARRAY_ALL];//最终的输出向量
				for (int ii = 0; ii < BLOCK_W_NVM; ii++)
				{
					for (int jj = 0; jj < BLOCK_H_NVM; jj++)//对于每个block进行操作
					{
						for (int kk = 0; kk < BIN_NVM; kk++)
						{
							f_Ether_Block[0][kk] = f_bin_out[ii*CELL_W_NVM + jj][kk];//block的左上角的cell
							f_Ether_Block[1][kk] = f_bin_out[ii*CELL_W_NVM + jj + 1][kk];//block的右上角的cell
							f_Ether_Block[2][kk] = f_bin_out[ii*CELL_W_NVM + jj + CELL_W_NVM][kk];//block的左下角的cell
							f_Ether_Block[3][kk] = f_bin_out[ii*CELL_W_NVM + jj + CELL_W_NVM + 1][kk];//block的右下角的cell
						}

						for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++)//4次循环，block中4个cell
						{
							for (int mm = 0; mm < BIN_NVM; mm++)//9次循环，找到最大的bin
							{
								f_max = (f_Ether_Block[ss][mm] > f_max) ? f_Ether_Block[ss][mm] : f_max;
							}
						}

						for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++)//归一化每个block 
						{
							for (int mm = 0; mm < BIN_NVM; mm++)
							{
								f_Ether_Block[ss][mm] /= f_max;
								f_Last_Array[uu++] = f_Ether_Block[ss][mm] * 扩大因子;
							}
						}
					}
				}

				Mat 测试test(1, ARRAY_ALL, CV_32FC1, f_Last_Array);//标签数据转换
				predictResult = svm->predict(测试test);//开始预测
			}
			else
			{
				cout << "未检测到人脸" << endl;
				cv::putText(temp, "no face", cv::Point(20, 60), 3, 2, cvScalar(0, 0, 255));

			}
			if (predictResult == 200) {
				cv::putText(temp, "1angry", cv::Point(20, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "1angry" << endl;
			}
			if (predictResult == 250) {
				cv::putText(temp, "2disgust", cv::Point(20, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "2disgust" << endl;
			}
			if (predictResult == 300) {
				cv::putText(temp, "3fear", cv::Point(20, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "3fear" << endl;
			}
			if (predictResult == 350) {
				cv::putText(temp, "4happy", cv::Point(20, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "4happy" << endl;
			}
			if (predictResult == 400) {
				cv::putText(temp, "5sadness", cv::Point(20, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "5sadness" << endl;
			}
			if (predictResult == 450) {
				cv::putText(temp, "6surprise", cv::Point(20, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "6surprise" << endl;
			}
			if (predictResult == 500) {
				cv::putText(temp, "7neutral", cv::Point(20, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "7neutral" << endl;
			}

			t = (double)cv::getTickCount() - t;
			t1 = t / cv::getTickFrequency();
			fps = 1.0 / t1;
			sprintf(string, "%.2f", fps);      // 帧率保留两位小数
			std::string fpsString("FPS:");
			fpsString += string;                    // 在"FPS:"后加入帧率数值字符串
			printf("detection time = %g ms\n", t * 1000 / cv::getTickFrequency());//////////////////////测量结束
			cv::putText(temp, fpsString, cv::Point(20, 460), cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 255, 0));
			imshow("表情识别      ESC退出", temp);
		}
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



void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin)//输入的是cell左上角x,y，i_w是图片的宽度，Img_in是输入图片，fbin是输出的9维bin数组。
{
	memset(fbin, 0, 9 * sizeof(float));
	float f_x = 0.0f, f_y = 0.0f, f_Nvm = 0.0f, f_theta = 0.0f;//f_x，f_y水平和垂直，f_Nvm表示数量，f_theta表示计算角度
	for (int ii = i_y; ii < i_y + i_w; ii++)//
	{
		for (int jj = i_x; jj < i_x + i_w; jj++)//从第一个像素开始计算
		{
			uchar* pData = (uchar*)(Img_in->imageData + ii * Img_in->widthStep + jj);
			f_x = pData[1] - pData[-1];
			f_y = pData[Img_in->widthStep] - pData[-Img_in->widthStep];
			f_Nvm = pow(f_x*f_x + f_y*f_y, 0.5f); //求出幅值

			float fAngle = 90.0f;
			if (f_x == 0.0f)//90°特殊情况
			{
				if (f_y > 0)
				{
					fAngle = 90.0f;
				}
			}
			else if (f_y == 0.0f)//0度特殊情况
			{
				if (f_x > 0)
				{
					fAngle == 0.0f;
				}
				else if (f_x < 0)//180度特殊情况
				{
					fAngle == 180.0f;
				}
			}
			else
			{
				f_theta = atan(f_y / f_x); //// atan() 范围为 -Pi/2 到 pi/2 所有9个bin范围是 0~180°
				fAngle = (BIN_SIZE*BIN_NVM * f_theta) / PI;//转化成度数
			}

			if (fAngle < 0)
			{
				fAngle += 180;//转化成0-180°
			}

			int iWhichBin = fAngle / BIN_SIZE;//角度除以20，例如150/20=7.5 = 7 ，落在第七个bin。165/20=8.5=8，落在第八个bin
			fbin[iWhichBin] += f_Nvm;
		}
	}
}
