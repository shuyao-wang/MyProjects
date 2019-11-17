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
#include <math.h>


//#define NORM_WIDTH 126
//#define NORM_HEIGHT 126
#define �������� 1000
#define PI 3.14
#define BIN_SIZE 20//bin��20�㻮��
#define BIN_NVM 9//bin������9��
#define NORM_WIDTH 128//ͼƬ��Сת�������130
#define NORM_HEIGHT 128//ͼƬ��Сת�������82
#define CELL_SIZE 14//cellΪ8*8��С
#define BLOCK_SIZE 2//block�Ĵ�С2*2
//#define PIC_CELL_WH 50
#define CELL_W_NVM  ((NORM_WIDTH-2) / CELL_SIZE)//cell�ĺ�������8
#define CELL_H_NVM  ((NORM_HEIGHT-2) / CELL_SIZE)//cell����������8
#define BLOCK_W_NVM  (CELL_W_NVM - BLOCK_SIZE + 1)//block�ĺ�������7
#define BLOCK_H_NVM  (CELL_H_NVM - BLOCK_SIZE + 1)//block����������7
#define CELL_NVM (CELL_W_NVM * CELL_H_NVM)//cell����64
#define BLOCK_NVM (BLOCK_W_NVM * BLOCK_H_NVM)//block����49
#define ARRAY_ALL (BLOCK_W_NVM * BLOCK_H_NVM * BLOCK_SIZE * BLOCK_SIZE * BIN_NVM)//block����*2*2*9������������ά��1764

//#define ÿ�ֱ���������Ŀ 80
//#define ÿ�ֱ���������Ŀ 90
#define ÿ�ֱ���������Ŀ 24
//#define ÿ�ֱ���������Ŀ 24
#define �������� 7
//#define ��������ά�� 7*7*58

using namespace dlib;
using namespace std;
//using namespace cv;
using namespace cv::ml;

//char * filePath[7] = {  "D:\\�ҵ����ݿ�\\A_�ҵı������ݼ�\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						  "D:\\�ҵ����ݿ�\\A_�ҵı������ݼ�\\2disgust",
//						  "D:\\�ҵ����ݿ�\\A_�ҵı������ݼ�\\3fear",
//						  "D:\\�ҵ����ݿ�\\A_�ҵı������ݼ�\\4happy",
//						  "D:\\�ҵ����ݿ�\\A_�ҵı������ݼ�\\5sadness",
//						  "D:\\�ҵ����ݿ�\\A_�ҵı������ݼ�\\6surprise",
//						  "D:\\�ҵ����ݿ�\\A_�ҵı������ݼ�\\7neutral"
//						}; //����·��
//
char * filePath[7] = {    "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\1angry",   //ͼƬ���ݼ���Ŀ¼
						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\2disgust",
						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\3fear",
						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\4happy",
						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\5sadness",
						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\6surprise",
						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\7neutral"
						}; //����·��
//char * filePath[7] = {    "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe5_���۽�����֤\\5\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe5_���۽�����֤\\5\\2disgust",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe5_���۽�����֤\\5\\3fear",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe5_���۽�����֤\\5\\4happy",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe5_���۽�����֤\\5\\5sadness",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe5_���۽�����֤\\5\\6surprise",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe5_���۽�����֤\\5\\7neutral"
//						}; //����·��

//char * filePath[7] = {    "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\2disgust",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\3fear",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\4happy",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\5sadness",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\6surprise",
//						  "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe1\\7neutral"
//						}; //����·��

//char * filePath[7] = {  "D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\2disgust",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\3fear",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\4happy",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\5sadness",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\6surprise",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\7neutral"
//						}; //����·��
//
//char * filePath[7] =  { "D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\1angry",   //ͼƬ���ݼ���Ŀ¼
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\2disgust",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\3fear",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\4happy",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\5sadness",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\6surprise",
//						"D:\\�ҵ����ݿ�\\CK+new\\CK+�����޹�\\7neutral"
//						}; //����·��

string �ļ�������[7] = { "1angry","2disgust","3fear","4happy","5sadness","6surprise","7neutral" };  //�ɼ����ļ����ļ���
//string svmModelFilePath = "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\lbp_uniform_492\\Classifier.xml";
string svmModelFilePath = "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\hog_8��8\\Classifier.xml";
//string svmModelFilePath = "D:\\�ҵ����ݿ�\\A_�ҵı������ݼ�\\my_emotion\\Classifier.xml";
//string svmModelFilePath = "D:\\�ҵ����ݿ�\\CK+new\\CK+�����й�\\hog_8��8\\Classifier.xml";
//string svmModelFilePath = "D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe5_���۽�����֤\\5\\lbp_uniform_16\\Classifier.xml";
void getFiles(string path, std::vector<string>& files);//�ļ�����
void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin);//�������cell���Ͻ�x,y��i_w��ͼƬ�Ŀ�ȣ�Img_in������ͼƬ��fbin�������9άbin���顣


int main()
{
		double init = 0, face_t = 0, faceAll = 0, shape_t = 0, shapeAll = 0;//����ʱ��
		cout << "==================================���ڿ�ʼ�ɼ�����====================================" << endl;
		cout << "                                                                                      " << endl;
		cout << "                                                                                      " << endl;
		//Mat featureVectorsOfSample;	//����������������
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;

		init = (double)cv::getTickCount();///////////////////////////////////////ģ�ͼ���ʱ��
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		init = (double)cv::getTickCount() - init;//�������


		static float f_Last_Array[ÿ�ֱ���������Ŀ*��������][ARRAY_ALL];//���յ��������
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

				IplImage* srcImg = cvLoadImage(files[i].c_str(), 0);//�����ݿ��еõ���ԭʼͼ��1

				CvSize size = cvSize(1.8*pointdistance, 2 * pointdistance);//ѡ�������С
				cvSetImageROI(srcImg, cvRect(eyecenter_x - 0.9*pointdistance, eyecenter_y - 0.5*pointdistance, size.width, size.height));//����Դͼ��ROI
				IplImage* pDest = cvCreateImage(size, srcImg->depth, srcImg->nChannels);//�����հ׵�Ŀ��ͼ��
				cvCopy(srcImg, pDest); //����ͼ��srcImg->pDest
				cv::Mat srcImage_1 = cv::cvarrToMat(pDest);
				//cvNamedWindow("Result2", CV_WINDOW_AUTOSIZE);
				//cvShowImage("Result2", pDest);
				Mat srcImage(NORM_WIDTH, NORM_HEIGHT, srcImage_1.type());
				resize(srcImage_1, srcImage, srcImage.size(), 0, 0, INTER_LINEAR);//srcImage��ȡ�õ�130*130��С��ͼƬ
				//cvNamedWindow("Result3", CV_WINDOW_AUTOSIZE);
				//cv::imshow("Result3", srcImage);
				//cvWaitKey(0);
				printf("get feature...\n");
				LOG_INFO_SVM_TEST("get feature...");
///////////////////////////////////////////////////////////////////////////////////////////HOG
				IplImage* img = &IplImage(srcImage);//�õ�Ԥ����֮���ͼ��
				//IplImage *img1 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
				IplImage *img2 = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
				CvMat* mat = cvCreateMat(img->width, img->height, CV_32FC1);

				//// �Ҷ�ͼ
				//cvCvtColor(img, img1, CV_BGR2GRAY);
				//cvNamedWindow("GrayImage", CV_WINDOW_AUTOSIZE);
				//cvShowImage("GrayImage", img1); //�Ҷ�ͼ��ʾ
				//cvWaitKey(0);
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
								f_Last_Array[j*ÿ�ֱ���������Ŀ+i][uu++] = f_Ether_Block[ss][mm] * ��������;
							}
						}
					}
				}
				//cvReleaseImage(&img);
				////cvReleaseImage(&img1);
				//cvReleaseImage(&img2);
				//cvReleaseMat(&mat);
				cout << "��ɵ�" << i + 1 << "��HOG����������ȡ" << endl;
				cout << "��" << j + 1 << "��Ŀ¼����ʣ�" << (float(i + 1) / ÿ�ֱ���������Ŀ) * 100 << "%" << endl;

			}//20��ѭ��
		}//7��forѭ��

				cout << "����ģ�͵���ʱ��:" << init * 1000 / cv::getTickFrequency() << "ms" << endl;
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

		Mat ѵ��train(ÿ�ֱ���������Ŀ*��������, ARRAY_ALL, CV_32FC1, f_Last_Array);//��ǩ����ת��
		Mat ��ǩlabel(ÿ�ֱ���������Ŀ*��������, 1, CV_32SC1, ������ǩ);//��ǩ����ת��
		time1 = getTickCount();
		svm->train(ѵ��train, ROW_SAMPLE, ��ǩlabel);//SVMѵ����featureVectorsOfSampleΪ���������������classOfSampleΪ��ǩ
		time2 = getTickCount();
		printf("ѵ��ʱ��:%fms\n", (time2 - time1)*1000. / getTickFrequency());
		//LOG_INFO_SVM_TEST("ѵ��ʱ��:%f", (time2 - time1)*1000. / getTickFrequency());
		printf("training done!\n");
		//LOG_INFO_SVM_TEST("training done!");
		// save model
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