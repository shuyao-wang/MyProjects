#include"SvmTest.h"

SVMTest::SVMTest(const string &_trainDataFileList,
	const string &_testDataFileList,
	const string &_svmModelFilePath,
	const string &_predictResultFilePath,
	SVM::Types svmType, // See SVM::Types. Default value is SVM::C_SVC.
	SVM::KernelTypes kernel,
	double c, // For SVM::C_SVC, SVM::EPS_SVR or SVM::NU_SVR. Default value is 0.
	double coef,  // For SVM::POLY or SVM::SIGMOID. Default value is 0.
	double degree, // For SVM::POLY. Default value is 0.
	double gamma, // For SVM::POLY, SVM::RBF, SVM::SIGMOID or SVM::CHI2. Default value is 1.
	double nu,  // For SVM::NU_SVC, SVM::ONE_CLASS or SVM::NU_SVR. Default value is 0.
	double p // For SVM::EPS_SVR. Default value is 0.
) :
	trainDataFileList(_trainDataFileList),
	testDataFileList(_testDataFileList),
	svmModelFilePath(_svmModelFilePath),
	predictResultFilePath(_predictResultFilePath)
{
	// set svm param
	svm = SVM::create();
	svm->setC(c);
	svm->setCoef0(coef);
	svm->setDegree(degree);
	svm->setGamma(gamma);
	svm->setKernel(kernel);
	svm->setNu(nu);
	svm->setP(p);
	svm->setType(svmType);

	//svm->setTermCriteria(TermCriteria(TermCriteria::EPS, 1000, FLT_EPSILON)); // based on accuracy
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6)); // based on the maximum number of iterations
}

bool SVMTest::Initialize()
{
	// initialize log
	//InitializeLog("SVMTest");

	return true;

}

SVMTest::~SVMTest()
{
}

void SVMTest::Train()
{
	// 读入训练样本图片路径和类别
	std::vector<string> imagePaths;
	std::vector<int> imageClasses;
	string line;
	std::ifstream trainingData(trainDataFileList, ios::out);
	while (getline(trainingData, line))
	{
		if (line.empty())
			continue;

		stringstream stream(line);
		string imagePath, imageClass;
		stream >> imagePath;
		stream >> imageClass;

		imagePaths.push_back(imagePath);// 读入训练样本图片路径
		imageClasses.push_back(atoi(imageClass.c_str()));// 读入训练样本类别
	}
	trainingData.close();

	printf("%d\n", imagePaths.size());//统计训练样本数量

									  // extract feature
	Mat featureVectorsOfSample;	//建立特征向量矩阵
	Mat classOfSample;			//建立标签矩阵
	printf("get feature...\n");
	LOG_INFO_SVM_TEST("get feature...");
	for (int i = 0; i <= imagePaths.size() - 1; ++i)//此文件的路径
	{
		Mat srcImage = imread(imagePaths[i], -1);//读取训练样本图片
		if (srcImage.empty() || srcImage.depth() != CV_8U)
		{
			printf("%s srcImage.empty()||srcImage.depth()!=CV_8U!\n", imagePaths[i].c_str());
			LOG_ERROR_SVM_TEST("%s srcImage.empty()||srcImage.depth()!=CV_8U!", imagePaths[i].c_str());
			continue;
		}

		// extract feature，开始特取特征
		Mat featureVector;//新建特征向量
		LBP lbp;
		lbp.ComputeLBPFeatureVector_Rotation_Uniform(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVector);//提取特征向量，利用的旋转不变等价模式
		if (featureVector.empty())
			continue;

		featureVectorsOfSample.push_back(featureVector);//把每一个图片进行分块，LBP计算，特取特征向量之后存入featureVectorsOfSample
		classOfSample.push_back(imageClasses[i]);//打标签

		printf("get feature... %f% \n", (i + 1)*100.0 / imagePaths.size());
		LOG_INFO_SVM_TEST("get feature... %f", (i + 1)*100.0 / imagePaths.size());
	}

	printf("get feature done!\n");
	LOG_INFO_SVM_TEST("get feature done!");

	// train
	printf("training...\n");
	LOG_INFO_SVM_TEST("training...");
	double time1, time2;
	time1 = getTickCount();
	svm->train(featureVectorsOfSample, ROW_SAMPLE, classOfSample);//SVM训练，featureVectorsOfSample为输入的特征向量，classOfSample为标签
	time2 = getTickCount();
	printf("训练时间:%f\n", (time2 - time1)*1000. / getTickFrequency());
	LOG_INFO_SVM_TEST("训练时间:%f", (time2 - time1)*1000. / getTickFrequency());
	printf("training done!\n");
	LOG_INFO_SVM_TEST("training done!");

	// save model
	svm->save(svmModelFilePath);//模型存储
}

void SVMTest::Predict()
{
	// predict
	std::vector<string> testImagePaths;
	std::vector<int> testImageClasses;
	string line;
	std::ifstream testData(testDataFileList, ios::out);
	while (getline(testData, line))
	{
		if (line.empty())
			continue;

		stringstream stream(line);
		string imagePath, imageClass;
		stream >> imagePath;
		stream >> imageClass;

		testImagePaths.push_back(imagePath);
		testImageClasses.push_back(atoi(imageClass.c_str()));

	}
	testData.close();

	printf("predicting...\n");
	LOG_INFO_SVM_TEST("predicting...");

	int numberOfRight_0 = 0;
	int numberOfError_0 = 0;
	int numberOfRight_1 = 0;
	int numberOfError_1 = 0;

	std::ofstream fileOfPredictResult(predictResultFilePath, ios::out); //最后识别的结果
	double sum_Predict = 0, sum_ExtractFeature = 0;
	char line2[256] = { 0 };
	for (int i = 0; i < testImagePaths.size(); ++i)
	{
		Mat srcImage = imread(testImagePaths[i], -1);
		if (srcImage.empty() || srcImage.depth() != CV_8U)
		{
			printf("%s srcImage.empty()||srcImage.depth()!=CV_8U!\n", testImagePaths[i].c_str());
			LOG_ERROR_SVM_TEST("%s srcImage.empty()||srcImage.depth()!=CV_8U!", testImagePaths[i].c_str());
			continue;
		}

		// extract feature
		double time1_ExtractFeature = getTickCount();
		Mat featureVectorOfTestImage;
		LBP lbp;
		lbp.ComputeLBPFeatureVector_Rotation_Uniform(srcImage, Size(CELL_SIZE, CELL_SIZE), featureVectorOfTestImage);
		if (featureVectorOfTestImage.empty())
			continue;
		double time2_ExtractFeature = getTickCount();
		sum_ExtractFeature += (time2_ExtractFeature - time1_ExtractFeature) * 1000 / getTickFrequency();

		//对测试图片进行分类并写入文件
		double time1_Predict = getTickCount();
		int predictResult = svm->predict(featureVectorOfTestImage);//开始预测
		double time2_Predict = getTickCount();
		sum_Predict += (time2_Predict - time1_Predict) * 1000 / getTickFrequency();

		sprintf_s(line2, "%s %d\n", testImagePaths[i].c_str(), predictResult);
		fileOfPredictResult << line2;
		LOG_INFO_SVM_TEST("%s %d", testImagePaths[i].c_str(), predictResult);

		// 0
		if ((testImageClasses[i] == 0) && (predictResult == 0))
		{
			++numberOfRight_0;
		}
		if ((testImageClasses[i] == 0) && (predictResult != 0))
		{
			++numberOfError_0;
		}

		// 1
		if ((testImageClasses[i] == 1) && (predictResult == 1))
		{
			++numberOfRight_1;
		}
		if ((testImageClasses[i] == 1) && (predictResult != 1))
		{
			++numberOfError_1;
		}

		printf("predicting...%f%\n", 100.0*(i + 1) / testImagePaths.size());
	}
	printf("predicting done!\n");
	LOG_INFO_SVM_TEST("predicting done!");

	printf("extract feature time：%f\n", sum_ExtractFeature / testImagePaths.size());
	LOG_INFO_SVM_TEST("extract feature time：%f", sum_ExtractFeature / testImagePaths.size());
	sprintf_s(line2, "extract feature time：%f\n", sum_ExtractFeature / testImagePaths.size());
	fileOfPredictResult << line2;

	printf("predict time：%f\n", sum_Predict / testImagePaths.size());
	LOG_INFO_SVM_TEST("predict time：%f", sum_Predict / testImagePaths.size());
	sprintf_s(line2, "predict time：%f\n", sum_Predict / testImagePaths.size());
	fileOfPredictResult << line2;


	// 0
	double accuracy_0 = (100.0*(numberOfRight_0)) / (numberOfError_0 + numberOfRight_0);
	printf("0：%f\n", accuracy_0);
	LOG_INFO_SVM_TEST("0：%f", accuracy_0);
	sprintf_s(line2, "0：%f\n", accuracy_0);
	fileOfPredictResult << line2;

	// 1
	double accuracy_1 = (100.0*numberOfRight_1) / (numberOfError_1 + numberOfRight_1);
	printf("1：%f\n", accuracy_1);
	LOG_INFO_SVM_TEST("1：%f", accuracy_1);
	sprintf_s(line2, "1：%f\n", accuracy_1);
	fileOfPredictResult << line2;

	// accuracy
	double accuracy_All = (100.0*(numberOfRight_1 + numberOfRight_0)) / (numberOfError_0 + numberOfRight_0 + numberOfError_1 + numberOfRight_1);
	printf("accuracy：%f\n", accuracy_All);
	LOG_INFO_SVM_TEST("accuracy:%f", accuracy_All);
	sprintf_s(line2, "accuracy:%f\n", accuracy_All);
	fileOfPredictResult << line2;

	fileOfPredictResult.close();

}