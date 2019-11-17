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
#include <math.h>


//#define NORM_WIDTH 126
//#define NORM_HEIGHT 126
#define 扩大因子 1000
#define PI 3.14
#define BIN_SIZE 20//bin的20°划分
#define BIN_NVM 9//bin的数量9个
#define NORM_WIDTH 128//图片大小转换，宽度130
#define NORM_HEIGHT 128//图片大小转换，宽度82
#define CELL_SIZE 14//cell为8*8大小
#define BLOCK_SIZE 2//block的大小2*2
//#define PIC_CELL_WH 50
#define CELL_W_NVM  ((NORM_WIDTH-2) / CELL_SIZE)//cell的横向数量8
#define CELL_H_NVM  ((NORM_HEIGHT-2) / CELL_SIZE)//cell的纵向数量8
#define BLOCK_W_NVM  (CELL_W_NVM - BLOCK_SIZE + 1)//block的横向数量7
#define BLOCK_H_NVM  (CELL_H_NVM - BLOCK_SIZE + 1)//block的纵向数量7
#define CELL_NVM (CELL_W_NVM * CELL_H_NVM)//cell总数64
#define BLOCK_NVM (BLOCK_W_NVM * BLOCK_H_NVM)//block总数49
#define ARRAY_ALL (BLOCK_W_NVM * BLOCK_H_NVM * BLOCK_SIZE * BLOCK_SIZE * BIN_NVM)//block总数*2*2*9，特征向量的维数1764

//#define 每种表情样本数目 80
//#define 每种表情样本数目 90
#define 每种表情样本数目 24
//#define 每种表情样本数目 24
#define 表情种类 7
//#define 特征向量维数 7*7*58

using namespace dlib;
using namespace std;
//using namespace cv;
using namespace cv::ml;

//char * filePath[7] = {  "D:\\我的数据库\\A_我的表情数据集\\1angry",   //图片数据集的目录
//						  "D:\\我的数据库\\A_我的表情数据集\\2disgust",
//						  "D:\\我的数据库\\A_我的表情数据集\\3fear",
//						  "D:\\我的数据库\\A_我的表情数据集\\4happy",
//						  "D:\\我的数据库\\A_我的表情数据集\\5sadness",
//						  "D:\\我的数据库\\A_我的表情数据集\\6surprise",
//						  "D:\\我的数据库\\A_我的表情数据集\\7neutral"
//						}; //样本路径
//
char * filePath[7] = {    "D:\\我的数据库\\JAFFE数据库\\jaffe0\\1angry",   //图片数据集的目录
						  "D:\\我的数据库\\JAFFE数据库\\jaffe0\\2disgust",
						  "D:\\我的数据库\\JAFFE数据库\\jaffe0\\3fear",
						  "D:\\我的数据库\\JAFFE数据库\\jaffe0\\4happy",
						  "D:\\我的数据库\\JAFFE数据库\\jaffe0\\5sadness",
						  "D:\\我的数据库\\JAFFE数据库\\jaffe0\\6surprise",
						  "D:\\我的数据库\\JAFFE数据库\\jaffe0\\7neutral"
						}; //样本路径
//char * filePath[7] = {    "D:\\我的数据库\\JAFFE数据库\\jaffe5_五折交叉验证\\5\\1angry",   //图片数据集的目录
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe5_五折交叉验证\\5\\2disgust",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe5_五折交叉验证\\5\\3fear",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe5_五折交叉验证\\5\\4happy",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe5_五折交叉验证\\5\\5sadness",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe5_五折交叉验证\\5\\6surprise",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe5_五折交叉验证\\5\\7neutral"
//						}; //样本路径

//char * filePath[7] = {    "D:\\我的数据库\\JAFFE数据库\\jaffe1\\1angry",   //图片数据集的目录
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\2disgust",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\3fear",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\4happy",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\5sadness",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\6surprise",
//						  "D:\\我的数据库\\JAFFE数据库\\jaffe1\\7neutral"
//						}; //样本路径

//char * filePath[7] = {  "D:\\我的数据库\\CK+new\\CK+与人有关\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\2disgust",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\3fear",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\4happy",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\5sadness",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\6surprise",
//						"D:\\我的数据库\\CK+new\\CK+与人有关\\7neutral"
//						}; //样本路径
//
//char * filePath[7] =  { "D:\\我的数据库\\CK+new\\CK+与人无关\\1angry",   //图片数据集的目录
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\2disgust",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\3fear",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\4happy",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\5sadness",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\6surprise",
//						"D:\\我的数据库\\CK+new\\CK+与人无关\\7neutral"
//						}; //样本路径

string 文件夹名称[7] = { "1angry","2disgust","3fear","4happy","5sadness","6surprise","7neutral" };  //采集的文件的文件夹
//string svmModelFilePath = "D:\\我的数据库\\JAFFE数据库\\jaffe0\\lbp_uniform_492\\Classifier.xml";
string svmModelFilePath = "D:\\我的数据库\\JAFFE数据库\\jaffe0\\hog_8×8\\Classifier.xml";
//string svmModelFilePath = "D:\\我的数据库\\A_我的表情数据集\\my_emotion\\Classifier.xml";
//string svmModelFilePath = "D:\\我的数据库\\CK+new\\CK+与人有关\\hog_8×8\\Classifier.xml";
//string svmModelFilePath = "D:\\我的数据库\\JAFFE数据库\\jaffe5_五折交叉验证\\5\\lbp_uniform_16\\Classifier.xml";
void getFiles(string path, std::vector<string>& files);//文件遍历
void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin);//输入的是cell左上角x,y，i_w是图片的宽度，Img_in是输入图片，fbin是输出的9维bin数组。


int main()
{
		double init = 0, face_t = 0, faceAll = 0, shape_t = 0, shapeAll = 0;//建立时间
		cout << "==================================现在开始采集数据====================================" << endl;
		cout << "                                                                                      " << endl;
		cout << "                                                                                      " << endl;
		//Mat featureVectorsOfSample;	//建立特征向量矩阵
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;

		init = (double)cv::getTickCount();///////////////////////////////////////模型加载时间
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		init = (double)cv::getTickCount() - init;//测试完毕


		static float f_Last_Array[每种表情样本数目*表情种类][ARRAY_ALL];//最终的输出向量
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

				IplImage* srcImg = cvLoadImage(files[i].c_str(), 0);//从数据库中得到的原始图像1

				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//选择区域大小
				cvSetImageROI(srcImg, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//设置源图像ROI
				IplImage* pDest = cvCreateImage(size, srcImg->depth, srcImg->nChannels);//创建空白的目标图像
				cvCopy(srcImg, pDest); //复制图像：srcImg->pDest
				cv::Mat srcImage_1 = cv::cvarrToMat(pDest);
				//cvNamedWindow("Result2", CV_WINDOW_AUTOSIZE);
				//cvShowImage("Result2", pDest);
				Mat srcImage(NORM_WIDTH, NORM_HEIGHT, srcImage_1.type());
				resize(srcImage_1, srcImage, srcImage.size(), 0, 0, INTER_LINEAR);//srcImage截取得到130*130大小的图片
				//cvNamedWindow("Result3", CV_WINDOW_AUTOSIZE);
				//cv::imshow("Result3", srcImage);
				//cvWaitKey(0);
				printf("get feature...\n");
				LOG_INFO_SVM_TEST("get feature...");
///////////////////////////////////////////////////////////////////////////////////////////HOG
				IplImage* img = &IplImage(srcImage);//得到预处理之后的图像
				//IplImage *img1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
				IplImage *img2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
				CvMat* mat = cvCreateMat(img->width, img->height, CV_32FC1);

				//// 灰度图
				//cvCvtColor(img, img1, CV_BGR2GRAY);
				//cvNamedWindow("GrayImage", CV_WINDOW_AUTOSIZE);
				//cvShowImage("GrayImage", img1); //灰度图显示
				//cvWaitKey(0);
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
								f_Last_Array[j*每种表情样本数目+i][uu++] = f_Ether_Block[ss][mm] * 扩大因子;
							}
						}
					}
				}
				//cvReleaseImage(&img);
				////cvReleaseImage(&img1);
				//cvReleaseImage(&img2);
				//cvReleaseMat(&mat);
				cout << "完成第" << i + 1 << "个HOG特征向量提取" << endl;
				cout << "第" << j + 1 << "个目录完成率：" << (float(i + 1) / 每种表情样本数目) * 100 << "%" << endl;

			}//20次循环
		}//7次for循环

				cout << "加载模型的总时间:" << init * 1000 / cv::getTickFrequency() << "ms" << endl;
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

		Mat 训练train(每种表情样本数目*表情种类, ARRAY_ALL, CV_32FC1, f_Last_Array);//标签数据转换
		Mat 标签label(每种表情样本数目*表情种类, 1, CV_32SC1, 人脸标签);//标签数据转换
		time1 = getTickCount();
		svm->train(训练train, ROW_SAMPLE, 标签label);//SVM训练，featureVectorsOfSample为输入的特征向量，classOfSample为标签
		time2 = getTickCount();
		printf("训练时间:%fms\n", (time2 - time1)*1000. / getTickFrequency());
		//LOG_INFO_SVM_TEST("训练时间:%f", (time2 - time1)*1000. / getTickFrequency());
		printf("training done!\n");
		//LOG_INFO_SVM_TEST("training done!");
		// save model
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