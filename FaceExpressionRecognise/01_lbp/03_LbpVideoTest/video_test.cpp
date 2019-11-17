//#pragma comment(linker, "/subsystem:windows /entry:mainCRTStartup")//去除CMD窗口
#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  
#include "LBP.h"
#include "SvmTest.h"

#define NORM_WIDTH 140
#define NORM_HEIGHT 140

#define capwidth 640
#define caphight 480 

using namespace dlib;
using namespace std;
//using namespace cv;
using namespace cv::ml;

//string svmModelFilePath = "D:\\我的数据库\\JAFFE数据库\\jaffe1\\lbp_uniform_16\\Classifier.xml";
string svmModelFilePath = "D:\\我的数据库\\CK+new\\CK+与人无关\\lbp_uniform_49\\Classifier.xml";
//string svmModelFilePath = "D:\\我的数据库\\A_我的表情数据集\\my_emotion\\Classifier.xml";
int main()
{
	double face_t = 0, shape_t = 0;//建立时间
	double perTime = 0, perTimeReal=0;
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
		//image_window win;  

		// Load face detection and pose estimation models.  
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		cv::Ptr<SVM> svm = StatModel::load<SVM>(svmModelFilePath);
		//cv::Ptr<SVM> svm = StatModel::load<SVM>("D:\\我的数据库\\JAFFE数据库\\jaffe1\\lbp_uniform_16\\Classifier.xml");
		// Grab and process frames until the main window is closed by the user.  
		while (cv::waitKey(30) != 27)
		{
			// Grab a frame 
			perTime = (double)cv::getTickCount();//帧率测量
			cv::Mat temp,temp1;
			cap >> temp;//opencv画点的
			cap >> temp1;//无点的
			cv_image<bgr_pixel> cimg(temp);

			face_t = (double)cv::getTickCount();///////////////////////////////////////开始检测时间

			std::vector<dlib::rectangle> faces = detector(cimg);

			face_t = (double)cv::getTickCount() - face_t;////////////////////////////////人脸检测结束时间
			std::printf("face detection time = %g ms\n", face_t * 1000 / cv::getTickFrequency());

			shape_t = (double)cv::getTickCount();////////////////////////////////////////人脸定位开始时间
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));
			double eyecenter_x1;
			double eyecenter_y1;
			double eyecenter_x2;
			double eyecenter_y2;
			double eyecenter_x;
			double eyecenter_y;
			double pointdistance;
			int predictResult = 0;
			if (!shapes.empty()) {

				shape_t = (double)cv::getTickCount() - shape_t;
				std::printf("face loactaion time = %g ms\n", shape_t * 1000 / cv::getTickFrequency());//人脸定位结束时间

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
				for (int i = 0; i < 68; i++) {
					circle(temp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 2, cv::Scalar(255, 0, 0), -1);
					//  shapes[0].part(i).x();//68个  
				}

				cv::Mat dst;
				cvtColor(temp1, dst, CV_BGR2GRAY);//将摄像头得到的temp1图像转换成单通道图像
				IplImage* srcImg1 = &IplImage(dst);//从数据库中得到的原始图像1
				IplImage* srcImg = cvCreateImage(cvGetSize(srcImg1), IPL_DEPTH_8U, 1);//新建空白的与srcImg1大小一样的图片
				cvEqualizeHist(srcImg1, srcImg);//直方图均衡
				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//选择区域大小
				if ((eyecenter_x - 0.9*pointdistance > 0) && (eyecenter_y - 0.5*pointdistance > 0) && (eyecenter_x + 0.9*pointdistance < capwidth) && (eyecenter_y + 2*pointdistance < caphight))
				{
					cvSetImageROI(srcImg, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//设置源图像ROI
				}//判断是否超出范围
				else//超出范围则提示
				{
					cout << "超出范围！"<< endl;
					continue;
				}	
				IplImage* pDest = cvCreateImage(size, srcImg->depth, srcImg->nChannels);//创建空白的目标图像用于存放裁剪之后的图像
				cvCopy(srcImg, pDest); //复制图像：srcImg->pDest
				cvResetImageROI(srcImg);//释放原ROI

				cv::Mat srcImage_1 = cv::cvarrToMat(pDest);
				cv::Mat srcImage(NORM_WIDTH, NORM_HEIGHT, srcImage_1.type());

				resize(srcImage_1, srcImage, srcImage.size(), 0, 0, 1);//调整图像大小
				cv::imshow("Prepossess image", srcImage);//预处理之后的输出

				cv::Mat featureVectorOfTestImage;//新建特征向量
				LBP lbp;

				lbp.ComputeLBPFeatureVector_Uniform(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVectorOfTestImage);
				if (featureVectorOfTestImage.empty())
					continue;
				predictResult = svm->predict(featureVectorOfTestImage * 1000);//开始预测
				
			}
			else
			{
				cout << "未检测到人脸"<< endl;
				cv::putText(temp, "no face", cv::Point(200, 60), 3, 2, cvScalar(0, 0, 255));

			}
		
			if (predictResult == 200) {
				cv::putText(temp, "angry", cv::Point(200, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "angry" << endl;
			}
			if (predictResult == 250) {
				cv::putText(temp, "disgust", cv::Point(200, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "disgust" << endl;
			}
			if (predictResult == 300) {
				cv::putText(temp, "fear", cv::Point(200, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "fear" << endl;
			}
			if (predictResult == 350) {
				cv::putText(temp, "happy", cv::Point(200, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "happy" << endl;
			}
			if (predictResult == 400) {
				cv::putText(temp, "sadness", cv::Point(200, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "sadness" << endl;
			}
			if (predictResult == 450) {
				cv::putText(temp, "surprise", cv::Point(200, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "surprise" << endl;
			}
			if (predictResult == 500) {
				cv::putText(temp, "neutral", cv::Point(200, 60), 3, 2, cvScalar(0, 0, 255));
				cout << "neutral" << endl;
			}

			perTime = (double)cv::getTickCount() - perTime;
			perTimeReal = perTime / cv::getTickFrequency();
			fps = 1.0 / perTimeReal;
			sprintf(string, "%.2f", fps);      // 帧率保留两位小数
			std::string fpsString("FPS:");
			fpsString += string;                    // 在"FPS:"后加入帧率数值字符串
			printf("recognition time = %g ms\n", perTime * 1000 / cv::getTickFrequency());//////////////////////测量结束


			cv::putText(temp, "Face Detector/ms : " + to_string(face_t * 1000 / cv::getTickFrequency()), cv::Point(20, 420), cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 255, 0));
			cv::putText(temp, "Face Location/ms : " + to_string(shape_t * 1000 / cv::getTickFrequency()), cv::Point(20, 440), cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 255, 0));
			cv::putText(temp, fpsString, cv::Point(20, 460), cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 255, 0));
			imshow("表情识别      ESC退出", temp);
			
		}
	
}