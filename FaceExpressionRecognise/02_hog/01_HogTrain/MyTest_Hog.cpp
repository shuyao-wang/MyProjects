 #include <stdio.h>
 #include <cv.h>
 #include <cxcore.h>
 #include <highgui.h>
 #include <math.h>
#define PI 3.14
#define BIN_SIZE 20//bin的20°划分
#define BIN_NVM 9//bin的数量9个
#define NORM_WIDTH 130//图片大小转换，宽度130
#define NORM_HEIGHT 130//图片大小转换，宽度82
#define CELL_SIZE 8//cell为8*8大小
#define BLOCK_SIZE 2//block的大小2*2
#define PIC_CELL_WH 50
#define CELL_W_NVM  ((NORM_WIDTH-2) / CELL_SIZE)//cell的横向数量
#define CELL_H_NVM  ((NORM_HEIGHT-2) / CELL_SIZE)//cell的纵向数量
#define BLOCK_W_NVM  (CELL_W_NVM - BLOCK_SIZE + 1)//block的横向数量
#define BLOCK_H_NVM  (CELL_H_NVM - BLOCK_SIZE + 1)//block的纵向数量
#define CELL_NVM (CELL_W_NVM * CELL_H_NVM)//cell总数
#define BLOCK_NVM (BLOCK_W_NVM * BLOCK_H_NVM)//block总数
#define ARRAY_ALL (BLOCK_W_NVM * BLOCK_H_NVM * BLOCK_SIZE * BLOCK_SIZE * BIN_NVM)//block总数*2*2*9，特征向量的维数


void  func(int i_x, int i_y, int i_w, IplImage* Img_in, float* fbin)//输入的是cell左上角x,y，i_w是图片的宽度，Img_in是输入图片，fbin是输出的9维bin数组。
{
	memset(fbin, 0, 9*sizeof(float));
	float f_x = 0.0f, f_y = 0.0f, f_Nvm = 0.0f, f_theta = 0.0f;//f_x，f_y水平和垂直，f_Nvm表示数量，f_theta表示计算角度
	for (int ii = i_y; ii < i_y + i_w; ii++)//
	{
		for (int jj = i_x; jj < i_x + i_w; jj++)//从第一个像素开始计算
		{
			uchar* pData = (uchar*)(Img_in->imageData + ii * Img_in->widthStep + jj);
			f_x = pData[1] - pData[-1];
			f_y = pData[Img_in->widthStep]- pData[-Img_in->widthStep];
			f_Nvm = pow( f_x*f_x + f_y*f_y,  0.5f); //求出幅值

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
				f_theta = atan(f_y/f_x); //// atan() 范围为 -Pi/2 到 pi/2 所有9个bin范围是 0~180°
				fAngle = (BIN_SIZE*BIN_NVM * f_theta)/PI;//转化成度数
			}

			if (fAngle < 0)
			{
				fAngle += 180;//转化成0-180°
			}

			int iWhichBin = fAngle/BIN_SIZE;//角度除以20，例如150/20=7.5 = 7 ，落在第七个bin。165/20=8.5=8，落在第八个bin
			fbin[iWhichBin] += f_Nvm;
		}
	}
}
 
 void main()
 {
 	IplImage* img = cvLoadImage("D:\\我的数据库\\JAFFE数据库\\jaffe0\\1angry\\KA.AN1.39.tiff");//读取图像
 	IplImage *img1 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);//用来存放灰度图
 	IplImage *img2 = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
 	CvMat* mat = cvCreateMat(img->width, img->height,CV_32FC1);

	//// 灰度图
 	cvCvtColor(img,img1,CV_BGR2GRAY); 
 	cvNamedWindow("GrayImage",CV_WINDOW_AUTOSIZE);
 	cvShowImage("GrayImage",img1); //灰度图显示

	//// gamma校正
 	uchar* uData  = (uchar*)(img1->imageData);
 	float* fMat = mat->data.fl;
 
 	for (int ii = 0; ii < img1->imageSize; ii++)
 	{
 		fMat[ii] = pow( uData[ii],  0.5f); 
 		((uchar*)img2->imageData)[ii] = (uchar)(fMat[ii]);
 	} 

	//// 缩放原有图片
	IplImage* img3 = 0;
	CvSize dst_cvsize;
	dst_cvsize.width = NORM_WIDTH;
	dst_cvsize.height = NORM_HEIGHT;
	img3 = cvCreateImage(dst_cvsize, IPL_DEPTH_8U,1 );
	cvResize(img2, img3, CV_INTER_LINEAR);//img3是缩放之后的图片

	//// 计算每个cell每个梯度的大小和方向
	int i_binNvm = 0;
	float f_bin_out[CELL_NVM][BIN_NVM];//cell总数和bin的9个
	float i_AllbinNvm[][BLOCK_SIZE*BLOCK_SIZE*BIN_NVM] = {0.0f};//2*2*9=36
	int ii_nvm1 = 0, ii_nvm2 = 0;
	for (int ii = 1; ii + CELL_SIZE < img3->height; ii+=CELL_SIZE)
	{
		for (int jj = 1; jj + CELL_SIZE < img3->width; jj+=CELL_SIZE)
		{
			func(jj, ii, CELL_SIZE, img3, f_bin_out[i_binNvm++]);//输出f_bin_out结果为所有的bin对应的9维数据
		}
	}
	int iBlockWhichCell = 0;
	int uu = 0;
	float  f_max = 0.0f;
	float f_Ether_Block[BLOCK_SIZE*BLOCK_SIZE][BIN_NVM];//每一个block中的4个cell
	float f_Last_Array[ARRAY_ALL];//最终的输出向量
	for (int ii = 0; ii < BLOCK_W_NVM; ii++ )
	{
		for (int jj = 0; jj < BLOCK_H_NVM; jj++)//对于每个block进行操作
		{
			for (int kk = 0; kk < BIN_NVM; kk++ )
			{
				f_Ether_Block[0][kk] = f_bin_out[ii*CELL_W_NVM+jj][kk];//block的左上角的cell
				f_Ether_Block[1][kk] = f_bin_out[ii*CELL_W_NVM+jj+1][kk];//block的右上角的cell
				f_Ether_Block[2][kk] = f_bin_out[ii*CELL_W_NVM+jj+ CELL_W_NVM][kk];//block的左下角的cell
				f_Ether_Block[3][kk] = f_bin_out[ii*CELL_W_NVM+jj+ CELL_W_NVM+1][kk];//block的右下角的cell
			}

			for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++ )//4次循环，block中4个cell
			{
				for (int mm = 0; mm < BIN_NVM; mm++)//9次循环，找到最大的bin
				{
					f_max = (f_Ether_Block[ss][mm] > f_max) ? f_Ether_Block[ss][mm] : f_max;
				}
			}

			for (int ss = 0; ss < BLOCK_SIZE * BLOCK_SIZE; ss++ )//归一化每个block 
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

