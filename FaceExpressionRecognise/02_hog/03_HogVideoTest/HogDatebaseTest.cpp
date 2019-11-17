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

//#define NORM_WIDTH 126
//#define NORM_HEIGHT 126

#define capwidth 640
#define caphight 480

#define PI 3.14
#define BIN_SIZE 20//bin��20�㻮��
#define BIN_NVM 9//bin������9��
#define NORM_WIDTH 128//ͼƬ��Сת�������130
#define NORM_HEIGHT 128//ͼƬ��Сת�������82
//#define CELL_SIZE 8//cellΪ8*8��С
#define CELL_SIZE 14//cellΪ8*8��С
#define BLOCK_SIZE 2//block�Ĵ�С2*2
//#define PIC_CELL_WH 50
#define CELL_W_NVM  ((NORM_WIDTH-2) / CELL_SIZE)//cell�ĺ�������
#define CELL_H_NVM  ((NORM_HEIGHT-2) / CELL_SIZE)//cell����������
#define BLOCK_W_NVM  (CELL_W_NVM - BLOCK_SIZE + 1)//block�ĺ�������16-1
#define BLOCK_H_NVM  (CELL_H_NVM - BLOCK_SIZE + 1)//block����������
#define CELL_NVM (CELL_W_NVM * CELL_H_NVM)//cell����
#define BLOCK_NVM (BLOCK_W_NVM * BLOCK_H_NVM)//block����
#define ARRAY_ALL (BLOCK_W_NVM * BLOCK_H_NVM * BLOCK_SIZE * BLOCK_SIZE * BIN_NVM)//block����15*15*2*2*9������������ά��

#define �������� 1000

using namespace dlib;
using namespace std;
//using namespace cv;
using namespace cv::ml;
string svmModelFilePath = "D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\hog_8��8\\Classifier.xml";
void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin);//�������cell���Ͻ�x,y��i_w��ͼƬ�Ŀ�ȣ�Img_in������ͼƬ��fbin�������9άbin���顣

int main()
{

		double t = 0, t1 = 0;
		double fps;
		char string[10];  // ���ڴ��֡�ʵ��ַ���
		cv::VideoCapture cap(0);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			system("pause");
		}
		cap.set(CV_CAP_PROP_FRAME_WIDTH, capwidth);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, caphight);
		//float f_Last_Array[ARRAY_ALL];//���յ��������

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		cv::Ptr<SVM> svm = StatModel::load<SVM>(svmModelFilePath);//����xml�ļ�

		while (cv::waitKey(30) != 27)
		{
			cv::Mat temp, temp1;
			cap >> temp;
			cap >> temp1;
			t = (double)cv::getTickCount();///////////////////////////////////////��ʼ���ʱ��
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
				pointdistance = sqrt(pow(eyecenter_x1 - eyecenter_x2, 2) + pow(eyecenter_y1 - eyecenter_y2, 2));  //���������	
				for (int i = 0; i < 68; i++) {
					circle(temp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 2, cv::Scalar(255, 0, 0), -1);
					//  shapes[0].part(i).x();//68��  
				}
				cv::Mat dst;//
				cvtColor(temp1, dst, CV_BGR2GRAY);//������ͷ�õ���temp1ͼ��ת���ɵ�ͨ��ͼ��
				IplImage* srcImg = &IplImage(dst);//ת��Ϊ�Ҷ�ͼ
				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//ѡ�������С
				if ((eyecenter_x - 0.9*pointdistance > 0) && (eyecenter_y - 0.5*pointdistance > 0) && (eyecenter_x + 0.9*pointdistance < capwidth) && (eyecenter_y + 2 * pointdistance < caphight))
				{
					cvSetImageROI(srcImg, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//����Դͼ��ROI
				}//�ж��Ƿ񳬳���Χ
				else//������Χ����ʾ
				{
					cout << "������Χ��" << endl;
					continue;
				}
				IplImage* pDest = cvCreateImage(size, srcImg->depth, srcImg->nChannels);//�����հ׵�Ŀ��ͼ��
				cvCopy(srcImg, pDest); //����ͼ��srcImg->pDest
				cvResetImageROI(srcImg);//�ͷ�ԭROI
				cv::Mat srcImage_1 = cv::cvarrToMat(pDest);
				Mat srcImage(NORM_HEIGHT, NORM_WIDTH, srcImage_1.type());
				resize(srcImage_1, srcImage, srcImage.size(), 0, 0, INTER_LINEAR);
				cv::imshow("Result1", srcImage);//Ԥ����֮������

				float f_Last_Array[ARRAY_ALL];//���յ��������
				IplImage* img = &IplImage(srcImage);
				//IplImage *img1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
				IplImage *img2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
				CvMat* mat = cvCreateMat(img->width, img->height, CV_32FC1);
				//// gammaУ��
				uchar* uData = (uchar*)(img->imageData);
				float* fMat = mat->data.fl;

				for (int ii = 0; ii < img->imageSize; ii++)
				{
					fMat[ii] = pow(uData[ii], 0.5f);
					((uchar*)img2->imageData)[ii] = (uchar)(fMat[ii]);
				}
				//// ����ÿ��cellÿ���ݶȵĴ�С�ͷ���
				int i_binNvm = 0;
				float f_bin_out[CELL_NVM][BIN_NVM];//cell������bin��9��
				float i_AllbinNvm[][BLOCK_SIZE*BLOCK_SIZE*BIN_NVM] = { 0.0f };//2*2*9=36
				int ii_nvm1 = 0, ii_nvm2 = 0;
				for (int ii = 1; ii + CELL_SIZE < img2->height; ii += CELL_SIZE)
				{
					for (int jj = 1; jj + CELL_SIZE < img2->width; jj += CELL_SIZE)
					{
						func(jj, ii, CELL_SIZE, img2, f_bin_out[i_binNvm++]);//���f_bin_out���Ϊ���е�bin��Ӧ��9ά����
					}
				}
				int iBlockWhichCell = 0;
				int uu = 0;
				float  f_max = 0.0f;
				float f_Ether_Block[BLOCK_SIZE*BLOCK_SIZE][BIN_NVM];//ÿһ��block�е�4��cell
																	//float f_Last_Array[ÿ�ֱ���������Ŀ*��������][ARRAY_ALL];//���յ��������
				for (int ii = 0; ii < BLOCK_W_NVM; ii++)
				{
					for (int jj = 0; jj < BLOCK_H_NVM; jj++)//����ÿ��block���в���
					{
						for (int kk = 0; kk < BIN_NVM; kk++)
						{
							f_Ether_Block[0][kk] = f_bin_out[ii*CELL_W_NVM + jj][kk];//block�����Ͻǵ�cell
							f_Ether_Block[1][kk] = f_bin_out[ii*CELL_W_NVM + jj + 1][kk];//block�����Ͻǵ�cell
							f_Ether_Block[2][kk] = f_bin_out[ii*CELL_W_NVM + jj + CELL_W_NVM][kk];//block�����½ǵ�cell
							f_Ether_Block[3][kk] = f_bin_out[ii*CELL_W_NVM + jj + CELL_W_NVM + 1][kk];//block�����½ǵ�cell
						}

						for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++)//4��ѭ����block��4��cell
						{
							for (int mm = 0; mm < BIN_NVM; mm++)//9��ѭ�����ҵ�����bin
							{
								f_max = (f_Ether_Block[ss][mm] > f_max) ? f_Ether_Block[ss][mm] : f_max;
							}
						}

						for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++)//��һ��ÿ��block 
						{
							for (int mm = 0; mm < BIN_NVM; mm++)
							{
								f_Ether_Block[ss][mm] /= f_max;
								f_Last_Array[uu++] = f_Ether_Block[ss][mm] * ��������;
							}
						}
					}
				}

				Mat ����test(1, ARRAY_ALL, CV_32FC1, f_Last_Array);//��ǩ����ת��
				predictResult = svm->predict(����test);//��ʼԤ��
			}
			else
			{
				cout << "δ��⵽����" << endl;
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
			sprintf(string, "%.2f", fps);      // ֡�ʱ�����λС��
			std::string fpsString("FPS:");
			fpsString += string;                    // ��"FPS:"�����֡����ֵ�ַ���
			printf("detection time = %g ms\n", t * 1000 / cv::getTickFrequency());//////////////////////��������
			cv::putText(temp, fpsString, cv::Point(20, 460), cv::FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 255, 0));
			imshow("����ʶ��      ESC�˳�", temp);
		}
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



void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin)//�������cell���Ͻ�x,y��i_w��ͼƬ�Ŀ�ȣ�Img_in������ͼƬ��fbin�������9άbin���顣
{
	memset(fbin, 0, 9 * sizeof(float));
	float f_x = 0.0f, f_y = 0.0f, f_Nvm = 0.0f, f_theta = 0.0f;//f_x��f_yˮƽ�ʹ�ֱ��f_Nvm��ʾ������f_theta��ʾ����Ƕ�
	for (int ii = i_y; ii < i_y + i_w; ii++)//
	{
		for (int jj = i_x; jj < i_x + i_w; jj++)//�ӵ�һ�����ؿ�ʼ����
		{
			uchar* pData = (uchar*)(Img_in->imageData + ii * Img_in->widthStep + jj);
			f_x = pData[1] - pData[-1];
			f_y = pData[Img_in->widthStep] - pData[-Img_in->widthStep];
			f_Nvm = pow(f_x*f_x + f_y*f_y, 0.5f); //�����ֵ

			float fAngle = 90.0f;
			if (f_x == 0.0f)//90���������
			{
				if (f_y > 0)
				{
					fAngle = 90.0f;
				}
			}
			else if (f_y == 0.0f)//0���������
			{
				if (f_x > 0)
				{
					fAngle == 0.0f;
				}
				else if (f_x < 0)//180���������
				{
					fAngle == 180.0f;
				}
			}
			else
			{
				f_theta = atan(f_y / f_x); //// atan() ��ΧΪ -Pi/2 �� pi/2 ����9��bin��Χ�� 0~180��
				fAngle = (BIN_SIZE*BIN_NVM * f_theta) / PI;//ת���ɶ���
			}

			if (fAngle < 0)
			{
				fAngle += 180;//ת����0-180��
			}

			int iWhichBin = fAngle / BIN_SIZE;//�Ƕȳ���20������150/20=7.5 = 7 �����ڵ��߸�bin��165/20=8.5=8�����ڵڰ˸�bin
			fbin[iWhichBin] += f_Nvm;
		}
	}
}
