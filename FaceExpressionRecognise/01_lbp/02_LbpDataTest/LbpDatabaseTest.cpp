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

//
//#define ÿ�ֱ���������Ŀ 100
#define �������� 7
//#define ��������ά�� 7*7*58

using namespace dlib;
using namespace std;
//using namespace cv;
using namespace cv::ml;

char * filePath[7] = {  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\test\\1angry",   //ͼƬ���ݼ���Ŀ¼
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\test\\2disgust",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\test\\3fear",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\test\\4happy",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\test\\5sadness",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\test\\6surprise",
						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\test\\7neutral"
						}; //����·��
//char * filePath[7] = {  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\2disgust",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\3fear",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\4happy",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\5sadness",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\6surprise",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\7neutral"
//						}; //����·��
//char * filePath[7] = {  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\test\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\test\\2disgust",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\test\\3fear",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\test\\4happy",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\test\\5sadness",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\test\\6surprise",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\test\\7neutral"
//						}; //����·��
//char * filePath[7] = {  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\2disgust",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\3fear",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\4happy",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\5sadness",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\6surprise",
//						"D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\7neutral"
						//}; //����·��
//char * filePath[7] = { "D:\\�ҵ����ݿ�\\CK+new\\ck1\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\CK+new\\ck1\\2disgust",
//						"D:\\�ҵ����ݿ�\\CK+new\\ck1\\3fear",
//						"D:\\�ҵ����ݿ�\\CK+new\\ck1\\4happy",
//						"D:\\�ҵ����ݿ�\\CK+new\\ck1\\5sadness",
//						"D:\\�ҵ����ݿ�\\CK+new\\ck1\\6surprise",
//						"D:\\�ҵ����ݿ�\\CK+new\\ck1\\7neutral"
//						}; //����·��
//char * filePath[7] = { "D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\2disgust",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\3fear",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\4happy",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\5sadness",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\6surprise",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\7neutral"
//						}; //����·��
//char * filePath[7] = {  "D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\test\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\test\\2disgust",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\test\\3fear",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\test\\4happy",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\test\\5sadness",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\test\\6surprise",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\test\\7neutral"
//						}; //����·��
//
//char * filePath[7] = {  "D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\test\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\test\\2disgust",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\test\\3fear",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\test\\4happy",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\test\\5sadness",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\test\\6surprise",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\test\\7neutral"
//						}; //����·��

//char * filePath[7] = {	"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\1angry",   //ͼƬ���ݼ���Ŀ¼
//							"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\2disgust",
//							"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\3fear",
//							"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\4happy",
//							"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\5sadness",
//							"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\6surprise",
//							"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\7neutral"
//							}; //����·��


string ��������[7] = { "1angry","2disgust","3fear","4happy","5sadness","6surprise","7neutral" };  //�ɼ����ļ����ļ���
string svmModelFilePath = "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\lbp_uniform_49\\Classifier.xml";
//string svmModelFilePath = "D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\lbp_uniform_49\\Classifier.xml";

void getFiles(string path, std::vector<string>& files);//�ļ�����

int main()
{

		double init = 0, face_t = 0, faceAll = 0, shape_t = 0, shapeAll = 0;//����ʱ��
		double all_time = 0, per_time = 0;
		int all_number = 0;
		float ���ս��[7] = { 0,0,0,0,0,0,0 };
		std::cout << "��ʼ����" << endl;
		int result;

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;

		init = (double)cv::getTickCount();///////////////////////////////////////ģ�ͼ���ʱ��
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		init = (double)cv::getTickCount() - init;//�������

		cv::Ptr<SVM> svm = StatModel::load<SVM>(svmModelFilePath);//����xml�ļ�
		for (int j = 0; j < 7; j++)//�ֱ𽫲ɼ������ݴ���7���ļ�����
		{
			int ����[7] = { 0,0,0,0,0,0,0 };
			std::cout << "��ʼ���Ե�" << j + 1 << "���ļ�" << endl;
			std::vector<string> files;//�ļ�����
			getFiles(filePath[j], files);		//�ļ�����
			int number = files.size();			//��ȡ�ļ����в���ͼƬ������
			all_number = all_number + number;
			for (int i = 0; i < number; i++)	//�ɼ����ݿ�ʼ
			{

				cv::Mat temp = cv::imread(files[i].c_str());
				per_time = (double)cv::getTickCount();/////////////////////////////////////////������⿪ʼʱ��
				face_t = (double)cv::getTickCount();/////////////////////////////////////////������⿪ʼʱ��

				cv_image<bgr_pixel> cimg(temp);
				std::vector<dlib::rectangle> faces = detector(cimg);

				face_t = (double)cv::getTickCount() - face_t;////////////////////////////////����������ʱ��
				std::printf("face detection time = %g ms\n", face_t * 1000 / cv::getTickFrequency());
				faceAll = faceAll + face_t;
				face_t = 0;

				shape_t = (double)cv::getTickCount();////////////////////////////////////////������λ��ʼʱ��
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
				//���Ҷ�ͼ�����ֱ��ͼ����
				IplImage* srcImg_equalize = cvCreateImage(cvGetSize(srcImg_gary), IPL_DEPTH_8U, 1);
				cvEqualizeHist(srcImg_gary, srcImg_equalize);////////////////////////ֱ��ͼ����
				 //�����⻯���ͼ����вü����õ��ü�֮���ͼ��
				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//ѡ�������С
				cvSetImageROI(srcImg_equalize, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//����Դͼ��ROI
				IplImage* srcImg_ori = cvCreateImage(size, srcImg_equalize->depth, srcImg_equalize->nChannels);//�����հ׵�Ŀ��ͼ��
				cvCopy(srcImg_equalize, srcImg_ori); //����ͼ��srcImg->pDest
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
				//cv::Mat srcHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
				//cv::Mat dstHistImage = cv::Mat::zeros(256, 256, CV_8UC1);
				//drawHistImg(cv::cvarrToMat(srcImg_gary), srcHistImage, "srcHistImage");
				//drawHistImg(cv::cvarrToMat(srcImg_equalize), dstHistImage, "dstHistImage");
				//cvWaitKey(0);
				printf("lbp get feature...\n");
				Mat featureVectorOfTestImage;//�½���������
				LBP lbp;
				lbp.ComputeLBPFeatureVector_Uniform(cv::cvarrToMat(srcImg_resize), Size(CELL_SIZE, CELL_SIZE), featureVectorOfTestImage);
				if (featureVectorOfTestImage.empty())
					continue;
				int predictResult=svm->predict(featureVectorOfTestImage * 1000);//��ʼԤ��

				switch (predictResult)
				{
						case 200:����[0]++; break;
						case 250:����[1]++; break;
						case 300:����[2]++; break;
						case 350:����[3]++; break;
						case 400:����[4]++; break;
						case 450:����[5]++; break;
						case 500:����[6]++; break;
				}
				per_time = (double)cv::getTickCount() - per_time;////////////////////////////////����ʶ�����ʱ��
				all_time = all_time + per_time;
				printf("recognition time = %g ms per image\n", per_time * 1000 / cv::getTickFrequency());//////////////////////��������
				per_time = 0;

				std::cout <<  "�����:" << float(i+1)*100/number << "%" << endl;

			}//ÿһ�ֱ�����ļ����ļ�Ŀ¼��forѭ��
			std::cout << ��������[j] << "�к���:" << number << "�Ų���ͼƬ" << endl;
			for (int x = 0; x < 7; x++)
			{
				std::cout << ��������[x] << "��" << ����[x] << endl;
			}
			std::cout << "ʶ���ʣ�" << (float)����[j] / number << endl;
			���ս��[j] = (float)����[j] / number;

		}//7�ֱ����for
		for (int y = 0; y < 7; y++)
		{
			std::cout << ��������[y] << "��ʶ���ʣ�" << ���ս��[y] << endl;
		}
		std::cout << "ƽ��ʶ���ʣ�" << float(���ս��[0] + ���ս��[1] + ���ս��[2] + ���ս��[3] + ���ս��[4] + ���ս��[5] + ���ս��[6]) / 7 << endl;
		std::cout << "����ģ�͵���ʱ��:" << init * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "���������ʱ��:  " << faceAll * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "�������ƽ��ʱ��:" << faceAll * 1000 / cv::getTickFrequency() / all_number << "ms" << endl;
		std::cout << "������λ����ʱ��:" << shapeAll * 1000 / cv::getTickFrequency() << "ms" << endl;
		std::cout << "������λƽ��ʱ��:" << shapeAll * 1000 / cv::getTickFrequency() / all_number << "ms" << endl;
		std::cout << "����ʶ����ʱ��:  " << all_time * 1000 / cv::getTickFrequency()  << "ms" << endl;
		std::cout << "����ʶ��ƽ��ʱ��:" << all_time * 1000 / cv::getTickFrequency() / all_number << "ms" << endl;
		
		std::system("pause");



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
