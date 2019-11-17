//#pragma comment(linker, "/subsystem:windows /entry:mainCRTStartup")//ȥ��CMD����

#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  
#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>  
#include <io.h> //�����ļ���غ���
#include "LBP.h"
#include "SvmTest.h"

#define NORM_WIDTH 140
#define NORM_HEIGHT 140
#define ÿ�ֱ���������Ŀ 24
#define �������� 7

using namespace dlib;
using namespace std;
//using namespace cv;
using namespace cv::ml;

//
//char * filePath[7] = {  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\2disgust",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\3fear",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\4happy",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\5sadness",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\6surprise",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\7neutral"
//						}; //����·��

char * filePath[7] = {  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\1angry",   //ͼƬ���ݼ���Ŀ¼
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\2disgust",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\3fear",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\4happy",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\5sadness",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\6surprise",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\7neutral"
}; //����·��

//char * filePath[7] =  { "D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\2disgust",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\3fear",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\4happy",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\5sadness",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\6surprise",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\7neutral"
//						}; //����·��
//char * filePath[7] = {	"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\2disgust",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\3fear",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\4happy",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\5sadness",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\6surprise",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\7neutral"
//						}; //����·��
//char * filePath[7] = {	"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\2disgust",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\3fear",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\4happy",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\5sadness",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\6surprise",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\7neutral"
//						}; //����·��

string �ļ�������[7] = { "1angry","2disgust","3fear","4happy","5sadness","6surprise","7neutral" };  //�ɼ����ļ����ļ���
//string svmModelFilePath = "D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\lbp_uniform_49\\Classifier.xml";
string svmModelFilePath = "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\lbp_uniform_49\\Classifier.xml";
void getFiles(string path, std::vector<string>& files);//�ļ�����
void drawHistImg(cv::Mat &src, cv::Mat &histImage, std::string name);//ֱ��ͼ����
int main()
{

	double init = 0, face_t = 0, faceAll = 0, shape_t = 0, shapeAll = 0;//����ʱ��
	cout << "==================================���ڿ�ʼ�ɼ�����====================================" << endl;
	cout << "                                                                                      " << endl;
	cout << "                                                                                      " << endl;
	Mat featureVectorsOfSample;	//����������������
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;


	init = (double)cv::getTickCount();///////////////////////////////////////ģ�ͼ���ʱ��
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	init = (double)cv::getTickCount() - init;//�������


	for (int j = 0; j < 7; j++)//�ֱ𽫲ɼ������ݴ���7���ļ�����
	{
		cout << "%%%%%%%%%%%%%%%%%%%%%%%���ڲɼ�:" << �ļ�������[j] << "��������%%%%%%%%%%%%%%%%%%%%%%%" << endl;
		std::vector<string> files;
		getFiles(filePath[j], files);		//�ļ�����
		for (int i = 0; i < ÿ�ֱ���������Ŀ; i++)	//�ɼ����ݿ�ʼ
		{
			cv::Mat temp = cv::imread(files[i].c_str());

			face_t = (double)cv::getTickCount();/////////////////////////////////////////������⿪ʼʱ��

			cv_image<bgr_pixel> cimg(temp);
			std::vector<dlib::rectangle> faces = detector(cimg);

			face_t = (double)cv::getTickCount() - face_t;////////////////////////////////����������ʱ��
			std::printf("face detection time = %g ms\n", face_t * 1000 / cv::getTickFrequency());
			faceAll = faceAll + face_t;
			face_t = 0;

			std::vector<full_object_detection> shapes;

			shape_t = (double)cv::getTickCount();////////////////////////////////////////������λ��ʼʱ��
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
				std::printf("face loactaion time = %g ms\n", shape_t * 1000 / cv::getTickFrequency());//������λ����ʱ��
				shapeAll = shapeAll + shape_t;//������λ��ʱ��
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
				pointdistance = sqrt(pow(eyecenter_x1 - eyecenter_x2, 2) + pow(eyecenter_y1 - eyecenter_y2, 2));  //���������
			}
			//�����ݼ��ж�ȡͼƬ����ת��Ϊ�Ҷ�ͼ��
			IplImage* srcImg_gary = cvLoadImage(files[i].c_str(), 0);//�����ݿ��еõ���ԭʼͼ��1��0��ɫ��1��ɫ
			//cvNamedWindow("srcImg_gary", CV_WINDOW_AUTOSIZE);
			//cvShowImage("srcImg_gary", srcImg_gary);
			//���Ҷ�ͼ�����ֱ��ͼ����
			IplImage* srcImg_equalize = cvCreateImage(cvGetSize(srcImg_gary), IPL_DEPTH_8U, 1);
			cvEqualizeHist(srcImg_gary, srcImg_equalize);////////////////////////ֱ��ͼ����
			//cvNamedWindow("srcImg_equalize", CV_WINDOW_AUTOSIZE);
			//cvShowImage("srcImg_equalize", srcImg_equalize);//��һ����ͼ��
			//�����⻯���ͼ����вü����õ��ü�֮���ͼ��
			CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//ѡ�������С
			cvSetImageROI(srcImg_equalize, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//����Դͼ��ROI
			IplImage* srcImg_ori = cvCreateImage(size, srcImg_equalize->depth, srcImg_equalize->nChannels);//�����հ׵�Ŀ��ͼ��
			cvCopy(srcImg_equalize, srcImg_ori); //����ͼ��srcImg->pDest
			//cvNamedWindow("srcImg_ori", CV_WINDOW_AUTOSIZE);
			//cvShowImage("srcImg_ori", srcImg_ori);//��ԭʼ��RGBͼ��
			//cvWaitKey(0);
			//���ü�֮���ͼ����г߶ȹ�һ��
			IplImage* srcImg_resize;
			CvSize norm_cvsize;
			norm_cvsize.width = NORM_WIDTH;  //Ŀ��ͼ��Ŀ�    
			norm_cvsize.height = NORM_HEIGHT; //Ŀ��ͼ��ĸ�  
			srcImg_resize = cvCreateImage(norm_cvsize, srcImg_ori->depth, srcImg_ori->nChannels);//����Ŀ��ͼ��  
			cvResize(srcImg_ori, srcImg_resize, CV_INTER_LINEAR); //����Դͼ��Ŀ��ͼ�� 
			//cvNamedWindow("srcImg_resize", CV_WINDOW_AUTOSIZE);
			//cvShowImage("srcImg_resize", srcImg_resize);//��һ����ͼ��
			//cvWaitKey(0);
			////���ƻҶ�ֱ��ͼ
			//cv::Mat srcHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
			//cv::Mat dstHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
			//drawHistImg(cv::cvarrToMat(srcImg_gary), srcHistImage, "srcHistImage");
			//drawHistImg(cv::cvarrToMat(srcImg_equalize), dstHistImage, "dstHistImage");
			//cvWaitKey(0);
			printf("lbp get feature...\n");
			Mat featureVector;//ÿ����������������
			LBP lbp;
			lbp.ComputeLBPFeatureVector_Uniform(cv::cvarrToMat(srcImg_resize), Size(CELL_SIZE, CELL_SIZE), featureVector);
			if (featureVector.empty())
				continue;
			featureVectorsOfSample.push_back(featureVector);

			cout << "��ɵ�" << i + 1 << "��LBP����������ȡ" << endl;
			cout << "��" << j + 1 << "��Ŀ¼����ʣ�"<<(float(i+1)/ ÿ�ֱ���������Ŀ)*100<<"%" << endl;
		}
	}
			cout << "����ģ�͵���ʱ��:" << init * 1000 / cv::getTickFrequency() <<"ms"<< endl;
			cout << "���������ʱ��:  " << faceAll * 1000 / cv::getTickFrequency() << "ms" << endl;
			cout << "������λ����ʱ��:" << shapeAll * 1000 / cv::getTickFrequency() << "ms" << endl;
			cout << "�������ƽ��ʱ��:" << faceAll * 1000 / cv::getTickFrequency() / (7 * ÿ�ֱ���������Ŀ) << "ms" << endl;
			cout << "������λƽ��ʱ��:" << shapeAll * 1000 / cv::getTickFrequency() / (7 * ÿ�ֱ���������Ŀ) << "ms" << endl;
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

	int ������ǩ[ÿ�ֱ���������Ŀ*��������];	//203��������ǩ
	for (int j = 0; j < ��������; j++) {

		for (int i = 0; i < ÿ�ֱ���������Ŀ; i++) {   //Ϊ��ǩ��ֵ
			������ǩ[i + ÿ�ֱ���������Ŀ*j] = 200 + 50 * j;     //��ǩΪ200��250��300��350��400��450��500
		}
	}
	Mat ��ǩlabel(ÿ�ֱ���������Ŀ*��������, 1, CV_32SC1, ������ǩ);//��ǩ����ת��
	time1 = getTickCount();
	svm->train(featureVectorsOfSample * 1000, ROW_SAMPLE, ��ǩlabel);//SVMѵ����featureVectorsOfSampleΪ���������������classOfSampleΪ��ǩ
	time2 = getTickCount();
	printf("ѵ��ʱ��:%fms\n", (time2 - time1)*1000. / getTickFrequency());
	printf("training done!\n");
	svm->save(svmModelFilePath);//ģ�ʹ洢
	system("pause");
}

void getFiles(string path, std::vector<string>& files)	//�ļ���������
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

////#pragma comment(linker, "/subsystem:windows /entry:mainCRTStartup")//ȥ��CMD����
//
//#include <dlib/opencv.h>  
//#include <opencv2/opencv.hpp>  
//#include <dlib/image_processing/frontal_face_detector.h>  
//#include <dlib/image_processing/render_face_detections.h>  
//#include <dlib/image_processing.h>  
//#include <dlib/gui_widgets.h>  
//#include <io.h> //�����ļ���غ���
//#include "LBP.h"
//#include "SvmTest.h"
//
//#define NORM_WIDTH 140
//#define NORM_HEIGHT 140
//#define ÿ�ֱ���������Ŀ 90
//#define �������� 7
////#define ��������ά�� 7*7*58+4*4*58
//
//using namespace dlib;
//using namespace std;
////using namespace cv;
//using namespace cv::ml;
//
////
////char * filePath[7] = {    "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\6surprise",   //ͼƬ���ݼ���Ŀ¼
////						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\2disgust",
////						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\3fear",
////						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\4happy",
////						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\5sadness",
////						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\6surprise",
////						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\7neutral"
////						}; //����·��
//
////char * filePath[7] =  { "D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\1angry",   //ͼƬ���ݼ���Ŀ¼
////						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\2disgust",
////						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\3fear",
////						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\4happy",
////						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\5sadness",
////						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\6surprise",
////						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\7neutral"
////						}; //����·��
//char * filePath[7] = { "D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\1angry1",   //ͼƬ���ݼ���Ŀ¼
//"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\2disgust",
//"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\3fear",
//"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\4happy",
//"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\5sadness",
//"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\6surprise",
//"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\7neutral"
//}; //����·��
//
//string �ļ�������[7] = { "1angry","2disgust","3fear","4happy","5sadness","6surprise","7neutral" };  //�ɼ����ļ����ļ���
////string svmModelFilePath = "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\lbp_uniform_492\\Classifier.xml";
////string svmModelFilePath = "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\lbp_uniform_64\\Classifier.xml";
////string svmModelFilePath = "D:\\�ҵ����ݿ�\\A_�ҵı������ݼ�\\my_emotion\\Classifier.xml";
//string svmModelFilePath = "D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\lbp_uniform_49\\Classifier.xml";
//void getFiles(string path, std::vector<string>& files);//�ļ�����
//void drawHistImg(cv::Mat &src, cv::Mat &histImage, std::string name);//ֱ��ͼ����
//int main()
//{
//		double t = 0;
//		cout << "==================================���ڿ�ʼ�ɼ�����====================================" << endl;
//		cout << "                                                                                      " << endl;
//		cout << "                                                                                      " << endl;
//		Mat featureVectorsOfSample;	//����������������
//		frontal_face_detector detector = get_frontal_face_detector();
//		shape_predictor pose_model;
//		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
//		std::vector<dlib::rectangle> faces2;
//		for (int j = 0; j < 7; j++)//�ֱ𽫲ɼ������ݴ���7���ļ�����
//		{
//			cout << "%%%%%%%%%%%%%%%%%%%%%%%���ڲɼ�:" << �ļ�������[j] << "��������%%%%%%%%%%%%%%%%%%%%%%%" << endl;
//			std::vector<string> files;
//			getFiles(filePath[j], files);		//�ļ�����
//			for (int i = 0; i < ÿ�ֱ���������Ŀ; i++)	//�ɼ����ݿ�ʼ
//			{
//				t = (double)cv::getTickCount();///////////////////////////////////////��ʼ���ʱ��
//				cv::Mat temp = cv::imread(files[i].c_str());
//				cv_image<bgr_pixel> cimg(temp);
//				std::vector<dlib::rectangle> faces = detector(cimg);
//				if (!faces.empty())faces2 = faces;
//				//t = (double)cv::getTickCount() - t;
//				//printf("detection time = %g ms\n", t * 1000 / cv::getTickFrequency());//////////////////////��������
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
//					pointdistance = sqrt(pow(eyecenter_x1 - eyecenter_x2, 2) + pow(eyecenter_y1 - eyecenter_y2, 2));  //���������
//					for (int i = 0; i < 68; i++) {
//						circle(temp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
//					}
//				}
//				//cv::imshow("�����ؼ��㶨λ",temp);
//				//waitKey(0);
//				IplImage* srcImg = cvLoadImage(files[i].c_str(), 0);//�����ݿ��еõ���ԭʼͼ��1
//				//IplImage* srcImg = cvCreateImage(cvGetSize(srcImg1), IPL_DEPTH_8U, 1);
//				//cvEqualizeHist(srcImg1, srcImg);////////////////////////ֱ��ͼ����
//				//cvNamedWindow("Result1", CV_WINDOW_AUTOSIZE);
//				//cvShowImage("Result1", srcImg);
//				//CvSize size = cvSize(faces[0].right() - faces[0].left(), faces[0].bottom() - faces[0].top());//ѡ�������С
//				//cvSetImageROI(srcImg, cvRect(faces[0].left(), faces[0].top(), size.width, size.height));//����Դͼ��ROI
//				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//ѡ�������С
//				cvSetImageROI(srcImg, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//����Դͼ��ROI
//				IplImage* pDest = cvCreateImage(size, srcImg->depth, srcImg->nChannels);//�����հ׵�Ŀ��ͼ��
//				cvCopy(srcImg, pDest); //����ͼ��srcImg->pDest
//				cv::Mat srcImage_1 = cv::cvarrToMat(pDest);
//				//cvNamedWindow("Result2", CV_WINDOW_AUTOSIZE);
//				//cvShowImage("Result2", pDest);
//				Mat srcImage(NORM_WIDTH, NORM_HEIGHT, srcImage_1.type());
//				resize(srcImage_1, srcImage, srcImage.size(), 0, 0, INTER_LINEAR);//˫���Բ�ֵ�㷨
//
//
//				cvNamedWindow("�Ҷ�ͼ", CV_WINDOW_AUTOSIZE);
//				cv::imshow("�Ҷ�ͼ", srcImage);
//				cvWaitKey(0);
//
//				//IplImage* srcImg1= &IplImage(srcImage);
//				IplImage* srcImg1 = cvCreateImage(cvGetSize(&IplImage(srcImage)), IPL_DEPTH_8U, 1);
//				cvEqualizeHist(&IplImage(srcImage), srcImg1);////////////////////////ֱ��ͼ����
//				cvNamedWindow("ֱ��ͼ����", CV_WINDOW_AUTOSIZE);
//				cvShowImage("ֱ��ͼ����", srcImg1);
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
//				Mat featureVector;//�½���������
//				LBP lbp;
//				//lbp.ComputeLBPFeatureVector_Rotation_Uniform(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVector);//��ȡ�������������õ���ת����ȼ�ģʽ
//				lbp.ComputeLBPFeatureVector_Uniform(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVector);
//				//lbp.ComputeLBPFeatureVector_Uniform(srcImage, Size(42, 42), featureVector2);
//				//cv::hconcat(featureVector1, featureVector2, featureVector);
//				if (featureVector.empty())
//					continue;
//				featureVectorsOfSample.push_back(featureVector);
//				//featureVectorsOfSample.push_back(featureVector1);
//				cout << "��ɴ��ǩ" << i + 1 << "�������ļ�" << endl;
//				cout << "��ɴ��ǩ" << j + 1 << "�����ݵĶ���" << endl;
//				t = (double)cv::getTickCount() - t;
//				printf("detection time = %g ms\n", t * 1000 / cv::getTickFrequency());//////////////////////��������
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
//		int ������ǩ[ÿ�ֱ���������Ŀ*��������];	//203��������ǩ
//		for (int j = 0; j < ��������; j++) {
//
//			for (int i = 0; i < ÿ�ֱ���������Ŀ; i++) {   //Ϊ��ǩ��ֵ
//				������ǩ[i + ÿ�ֱ���������Ŀ*j] = 200 + 50 * j;     //��ǩΪ200��250��300��350��400��450��500
//			}
//		}
//		Mat ��ǩlabel(ÿ�ֱ���������Ŀ*��������, 1, CV_32SC1, ������ǩ);//��ǩ����ת��
//		time1 = getTickCount();
//		svm->train(featureVectorsOfSample * 1000, ROW_SAMPLE, ��ǩlabel);//SVMѵ����featureVectorsOfSampleΪ���������������classOfSampleΪ��ǩ
//		time2 = getTickCount();
//		printf("ѵ��ʱ��:%f\n", (time2 - time1)*1000. / getTickFrequency());
//		LOG_INFO_SVM_TEST("ѵ��ʱ��:%f", (time2 - time1)*1000. / getTickFrequency());
//		printf("training done!\n");
//		LOG_INFO_SVM_TEST("training done!");
//		// save model
//		svm->save(svmModelFilePath);//ģ�ʹ洢
//}
//
//void getFiles(string path, std::vector<string>& files)	//�ļ���������
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
