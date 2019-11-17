 #include <stdio.h>
 #include <cv.h>
 #include <cxcore.h>
 #include <highgui.h>
 #include <math.h>
#define PI 3.14
#define BIN_SIZE 20//bin��20�㻮��
#define BIN_NVM 9//bin������9��
#define NORM_WIDTH 130//ͼƬ��Сת�������130
#define NORM_HEIGHT 130//ͼƬ��Сת�������82
#define CELL_SIZE 8//cellΪ8*8��С
#define BLOCK_SIZE 2//block�Ĵ�С2*2
#define PIC_CELL_WH 50
#define CELL_W_NVM  ((NORM_WIDTH-2) / CELL_SIZE)//cell�ĺ�������
#define CELL_H_NVM  ((NORM_HEIGHT-2) / CELL_SIZE)//cell����������
#define BLOCK_W_NVM  (CELL_W_NVM - BLOCK_SIZE + 1)//block�ĺ�������
#define BLOCK_H_NVM  (CELL_H_NVM - BLOCK_SIZE + 1)//block����������
#define CELL_NVM (CELL_W_NVM * CELL_H_NVM)//cell����
#define BLOCK_NVM (BLOCK_W_NVM * BLOCK_H_NVM)//block����
#define ARRAY_ALL (BLOCK_W_NVM * BLOCK_H_NVM * BLOCK_SIZE * BLOCK_SIZE * BIN_NVM)//block����*2*2*9������������ά��


void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin)//�������cell���Ͻ�x,y��i_w��ͼƬ�Ŀ�ȣ�Img_in������ͼƬ��fbin�������9άbin���顣
{
	memset(fbin, 0, 9*sizeof(float));
	float f_x = 0.0f, f_y = 0.0f, f_Nvm = 0.0f, f_theta = 0.0f;//f_x��f_yˮƽ�ʹ�ֱ��f_Nvm��ʾ������f_theta��ʾ����Ƕ�
	for (int ii = i_y; ii < i_y + i_w; ii++)//
	{
		for (int jj = i_x; jj < i_x + i_w; jj++)//�ӵ�һ�����ؿ�ʼ����
		{
			uchar* pData = (uchar*)(Img_in->imageData + ii * Img_in->widthStep + jj);
			f_x = pData[1] - pData[-1];
			f_y = pData[Img_in->widthStep]- pData[-Img_in->widthStep];
			f_Nvm = pow( f_x*f_x + f_y*f_y,  0.5f); //�����ֵ

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
				f_theta = atan(f_y/f_x); //// atan() ��ΧΪ -Pi/2 �� pi/2 ����9��bin��Χ�� 0~180��
				fAngle = (BIN_SIZE*BIN_NVM * f_theta)/PI;//ת���ɶ���
			}

			if (fAngle < 0)
			{
				fAngle += 180;//ת����0-180��
			}

			int iWhichBin = fAngle/BIN_SIZE;//�Ƕȳ���20������150/20=7.5 = 7 �����ڵ��߸�bin��165/20=8.5=8�����ڵڰ˸�bin
			fbin[iWhichBin] += f_Nvm;
		}
	}
}
 
 void main()
 {
 	IplImage* img = cvLoadImage("D:\\�ҵ����ݿ�\\JAFFE���ݿ�\\jaffe0\\1angry\\KA.AN1.39.tiff");//��ȡͼ��
 	IplImage *img1 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);//������ŻҶ�ͼ
 	IplImage *img2 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
 	CvMat* mat = cvCreateMat(img->width, img->height,CV_32FC1);

	//// �Ҷ�ͼ
 	cvCvtColor(img,img1,CV_BGR2GRAY); 
 	cvNamedWindow("GrayImage",CV_WINDOW_AUTOSIZE);
 	cvShowImage("GrayImage",img1); //�Ҷ�ͼ��ʾ

	//// gammaУ��
 	uchar* uData  = (uchar*)(img1->imageData);
 	float* fMat = mat->data.fl;
 
 	for (int ii = 0; ii < img1->imageSize; ii++)
 	{
 		fMat[ii] = pow( uData[ii],  0.5f); 
 		((uchar*)img2->imageData)[ii] = (uchar)(fMat[ii]);
 	} 

	//// ����ԭ��ͼƬ
	IplImage* img3 = 0;
	CvSize dst_cvsize;
	dst_cvsize.width = NORM_WIDTH;
	dst_cvsize.height = NORM_HEIGHT;
	img3 = cvCreateImage(dst_cvsize, IPL_DEPTH_8U,1 );
	cvResize(img2, img3, CV_INTER_LINEAR);//img3������֮���ͼƬ

	//// ����ÿ��cellÿ���ݶȵĴ�С�ͷ���
	int i_binNvm = 0;
	float f_bin_out[CELL_NVM][BIN_NVM];//cell������bin��9��
	float i_AllbinNvm[][BLOCK_SIZE*BLOCK_SIZE*BIN_NVM] = {0.0f};//2*2*9=36
	int ii_nvm1 = 0, ii_nvm2 = 0;
	for (int ii = 1; ii + CELL_SIZE < img3->height; ii+=CELL_SIZE)
	{
		for (int jj = 1; jj + CELL_SIZE < img3->width; jj+=CELL_SIZE)
		{
			func(jj, ii, CELL_SIZE, img3, f_bin_out[i_binNvm++]);//���f_bin_out���Ϊ���е�bin��Ӧ��9ά����
		}
	}
	int iBlockWhichCell = 0;
	int uu = 0;
	float  f_max = 0.0f;
	float f_Ether_Block[BLOCK_SIZE*BLOCK_SIZE][BIN_NVM];//ÿһ��block�е�4��cell
	float f_Last_Array[ARRAY_ALL];//���յ��������
	for (int ii = 0; ii < BLOCK_W_NVM; ii++ )
	{
		for (int jj = 0; jj < BLOCK_H_NVM; jj++)//����ÿ��block���в���
		{
			for (int kk = 0; kk < BIN_NVM; kk++ )
			{
				f_Ether_Block[0][kk] = f_bin_out[ii*CELL_W_NVM+jj][kk];//block�����Ͻǵ�cell
				f_Ether_Block[1][kk] = f_bin_out[ii*CELL_W_NVM+jj+1][kk];//block�����Ͻǵ�cell
				f_Ether_Block[2][kk] = f_bin_out[ii*CELL_W_NVM+jj+ CELL_W_NVM][kk];//block�����½ǵ�cell
				f_Ether_Block[3][kk] = f_bin_out[ii*CELL_W_NVM+jj+ CELL_W_NVM+1][kk];//block�����½ǵ�cell
			}

			for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++ )//4��ѭ����block��4��cell
			{
				for (int mm = 0; mm < BIN_NVM; mm++)//9��ѭ�����ҵ�����bin
				{
					f_max = (f_Ether_Block[ss][mm] > f_max) ? f_Ether_Block[ss][mm] : f_max;
				}
			}

			for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++ )//��һ��ÿ��block 
			{
				for (int mm = 0; mm < BIN_NVM; mm++)
				{
					f_Ether_Block[ss][mm] /= f_max;
					f_Last_Array[uu++] = f_Ether_Block[ss][mm]*1000;
				}
			}
		}
	}
 	cvReleaseImage(&img);
 	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	cvReleaseImage(&img3);
	//cvReleaseImage(&img4);
 	cvReleaseMat(&mat);
	//cvDestroyWindow("GrayImage");
	cvDestroyWindow("LineShow");
 }

