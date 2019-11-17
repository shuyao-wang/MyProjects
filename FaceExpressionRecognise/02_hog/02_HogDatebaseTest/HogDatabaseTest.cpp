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

//#define 每种表情样本数目 10
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
//						}; //样本路径


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

//char * filePath[7] = {      "D:\\我的数据库\\CK+new\\CK+与人无关\\1angry",   //图片数据集的目录
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\2disgust",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\3fear",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\4happy",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\5sadness",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\6surprise",
//							"D:\\我的数据库\\CK+new\\CK+与人无关\\7neutral"
//							}; //样本路径

string 表情名称[7] = { "1angry","2disgust","3fear","4happy","5sadness","6surprise","7neutral" };  //采集的文件的文件夹

//string svmModelFilePath = "D:\\我的数据库\\CK+new\\CK+与人有关\\hog_8×8\\Classifier.xml";
string svmModelFilePath = "D:\\我的数据库\\JAFFE数据库\\jaffe0\\hog_8×8\\Classifier.xml";

void getFiles(string path, std::vector<string>& files);//文件遍历
void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin);//输入的是cell左上角x,y，i_w是图片的宽度，Img_in是输入图片，fbin是输出的9维bin数组。

int main()
{

		double init = 0, face_t = 0, faceAll = 0, shape_t = 0, shapeAll = 0;//建立时间
		double all_time = 0, per_time = 0;
		int all_number = 0;
		float f_Last_Array[ARRAY_ALL];//最终的输出向量
		float 最终结果[7] = { 0,0,0,0,0,0,0 };
		cout << "开始测试" << endl;
		int result;
		double t = 0;
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;

		init = (double)cv::getTickCount();///////////////////////////////////////模型加载时间
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		init = (double)cv::getTickCount() - init;//测试完毕

		cv::Ptr<SVM> svm = StatModel::load<SVM>(svmModelFilePath);//加载xml文件
		for (int j = 0; j < 7; j++)//分别将采集的数据存入7个文件夹中
		{
			int 表情[7] = { 0,0,0,0,0,0,0 };
			cout << "开始测试第" << j + 1 << "个文件" << endl;
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

				
				IplImage* srcImg = cvLoadImage(files[i].c_str(), 0);//从数据库中得到的原始图像1
				
				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//选择区域大小
				cvSetImageROI(srcImg, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//设置源图像ROI
				IplImage* pDest = cvCreateImage(size, srcImg->depth, srcImg->nChannels);//创建空白的目标图像
				cvCopy(srcImg, pDest); //复制图像：srcImg->pDest
				cv::Mat srcImage_1 = cv::cvarrToMat(pDest);

				Mat srcImage(NORM_WIDTH, NORM_HEIGHT, srcImage_1.type());
				resize(srcImage_1, srcImage, srcImage.size(), 0, 0, INTER_LINEAR);
	
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
				printf("get feature...\n");
				Mat 测试test(1, ARRAY_ALL, CV_32FC1, f_Last_Array);//标签数据转换
				int predictResult=svm->predict(测试test);//开始预测

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
				cout <<  "完成率:" << float(i+1)*100/number << "%" << endl;

			}//每一种表情的文件夹文件目录下for循环
			cout << 表情名称[j] << "中含有:" << number << "张测试图片" << endl;
			for (int x = 0; x < 7; x++)
			{
				cout << 表情名称[x] << "：" << 表情[x] << endl;
			}
			cout << "识别率：" << (float)表情[j] / number << endl;
			最终结果[j] = (float)表情[j] / number;

		}//7种表情的for
		for (int y = 0; y < 7; y++)
		{
			cout << 表情名称[y] << "的识别率：" << 最终结果[y] << endl;
		}
		std::cout << "平均识别率：" << float(最终结果[0] + 最终结果[1] + 最终结果[2] + 最终结果[3] + 最终结果[4] + 最终结果[5] + 最终结果[6]) / 7 << endl;
		std::cout << "加载模型的总时间:" << init * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "人脸检测总时间:  " << faceAll * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "人脸检测平均时间:" << faceAll * 1000 / cv::getTickFrequency() / all_number << "ms" << endl;
		std::cout << "人脸定位的总时间:" << shapeAll * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "人脸定位平均时间:" << shapeAll * 1000 / cv::getTickFrequency() / all_number << "ms" << endl;
		std::cout << "表情识别总时间:  " << all_time * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "表情识别平均时间:" << all_time * 1000 / cv::getTickFrequency() / all_number << "ms" << endl;
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
