/**
  **************************************************************
  * @file       LPR.cpp
  * @author     ������
  * @version    V0.3.2
  * @date       2016-01-04
  *
  * @brief      ����ʶ������㷨����
  *
  * @details 
  * @verbatim
  * ����ʶ������㷨����
  *
  * �޸ļ�¼��
  * 2015-12-8 :
  *   - File Created.
  *
  * @endverbatim
  ***************************************************************
  */

#include "LPR.h"
#include "PlateImg.h"

using namespace cv;
using namespace std;

LPR::LPR()
{
}

LPR::~LPR()
{
}

/**
  * @brief  ��ֵ��ͼ��
  *
  * @param  Img: ������ͼ��ʹ�����ý��д���
  *
  * @retval None
  */
void LPR::binary(cv::Mat & Img)
{
  // ��ֵ�˲�
  medianBlur(Img, Img, 7);

  // ��ȡBͨ��תΪ�Ҷ�ͼ
  vector<Mat> mv;
  split(Img, mv);
  Img = mv[2];

  // ����ƽ���Ҷ�ֵ
  uchar ThreVal = (uchar)cv::mean(Img).val[0];

  // ��ֵ��
  threshold(Img, Img, ThreVal, 255, THRESH_BINARY);

  // ������̬ѧ�����븯ʴ������ȥ�����
  //Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
  //Mat element(2, 2, CV_8U, Scalar(1));
  //��ʴ
  erode(Img, Img, getStructuringElement(MORPH_RECT, Size(3, 3)));
  //����
  dilate(Img, Img, getStructuringElement(MORPH_RECT, Size(3, 3)));
  erode(Img, Img, getStructuringElement(MORPH_RECT, Size(2, 2)));

}

/**
  * @brief  �ݹ����ȡ��ֵ��ͼ���е�ÿ���ַ�
  *
  * @param  Img: ������Ķ�ֵ��ͼ��
  * @param  vec: ��һ������ַ�ͼ��vector���������������
  *
  * @retval None
  *
  * @note
  * ��������д��Ч�ʲ��ߣ����ҳ���Ҳ����࡭��
  */
void LPR::Extract(cv::Mat & Img, std::vector<cv::Mat> & vec)
{
  /* �ַ��߽� */
  int Left = 0, Right = Img.cols, Top = 0, Bottom = Img.rows;
  /* �����ۼӱ��� */
  int sum = 0;
  /* ��������������VerifyStart������ */
  int count = 0;
  const uchar CountMax = 3;

  /* ��ǰִ��״̬ */
  enum{ FindStart, VerifyStart, FindEnd, Exit } flag_r = FindStart, flag_c = FindStart;
  
  /* ����ɨ�裬ȷ�����ұ߽� */
  for (int j = 0; j < Img.cols; j++)
  {
    sum = 0;

    if (flag_c == Exit)
      break;

    /* �ۼ������� */
    for (int i = 0; i < Img.rows; i++)
    {
      sum += Img.at<uchar>(i, j);
    }

    switch (flag_c)
    {
    /* ��ʼ���ж���������һ�����ϰ�ɫ���ص� */
    case FindStart:
      if (sum > 255)
      {
        flag_c = VerifyStart;
        Left = j;
        count = 0;
      }
      break;

    /* ��ʼ�к�Ҫ������CountMax����һ�����ϰ�ɫ���ص���У������������ */
    case VerifyStart:
      if (count > CountMax)
        flag_c = FindEnd;
      else if (sum > 255)
        count++;
      else
        flag_c = FindStart;
      break;

    /* �������ж�����������һ�����°�ɫ���ص� */
    case FindEnd:
      if (sum <= 255)
      {
        Right = j;
        flag_c = Exit;
      }
      break;
    }
  }

  /* ����ɨ�裬ȷ�����±߽� */
  for (int i = 0; i < Img.rows; i++)
  {
    sum = 0;

    if (flag_r == Exit)
      break;

    /* �ۼ�Left~Right֮����� */
    /* ע�⣺�˴���Ҫ�ۼ������У�������ͼ������б��ʱ�����Ǵ���� */
    for (int j = Left; j < Right; j++)
    {
      sum += Img.at<uchar>(i, j);
    }

    switch (flag_r)
    {
      /* ��ʼ���ж���������һ�����ϰ�ɫ���ص� */
    case FindStart:
      if (sum > 255)
      {
        flag_r = VerifyStart;
        Top = i;
        count = 0;
      }
      break;

      /* ��ʼ�к�Ҫ������CountMax����һ�����ϰ�ɫ���ص���У������������ */
    case VerifyStart:
      if (count > CountMax)
        flag_r = FindEnd;
      else if (sum > 255)
        count++;
      else
        flag_r = FindStart;
      break;

      /* �������ж�����������һ�����°�ɫ���ص� */
    case FindEnd:
      if (sum <= 255)
      {
        Bottom = i;
        flag_r = Exit;
      }
      break;
    }
  }

  /* �ҵ���Ч�ַ� */
  if (flag_c == Exit || flag_c == FindEnd)
  {
    /* ���ַ�ͼƬ��С���й�һ������ */
    Mat tmp;
    resize(Img(Range(Top, Bottom), Range(Left, Right)), tmp, Size(CharImgWidth, CharImgHeight));

    /* ��һ������Ҫ���½��ж�ֵ�� */
    // ����ƽ���Ҷ�ֵ
    // ����ƽ���Ҷ�ֵ
    uchar ThreVal = (uchar)cv::mean(Img).val[0];
    // ��ֵ��
    threshold(tmp, tmp, ThreVal, 255, THRESH_BINARY);

    vec.push_back(tmp);
  }

  /* ���ҽ���flag_c == Exitʱ�ſ��ܻ��и�����ַ� */
  if (flag_c != Exit)
    return;
  else
    /* �ݹ顣ע�⣺��Ҫ�ü��У������ü��� */
    return Extract(Img(Range(0, Img.rows), Range(Right, Img.cols)), vec);
}

/**
  * @brief  ��ȡ��������
  *
  * @param  Img: ֮ǰ�ָ�õĹ�һ������ַ�ͼƬ
  *
  * @retval cv::Mat ��ȡ������������
  *
  */
cv::Mat LPR::Feature(cv::Mat Img)
{
  Mat w;
  switch (CurrentFeatureMethod)
  {
    /* SVD */
  case FeatureSVD:
    Img.convertTo(Img, CV_32F, 1 / 255.0);
    SVD::compute(Img, w);
    return w;

  case FeatureVec:
    w = Mat(Img.rows * Img.cols, 1, CV_32F);
    for (int i = 0; i < Img.rows; i++)
    {
      for (int j = 0; j < Img.cols; j++)
      {
        w.at<float>(i*Img.cols + j, 0) = (float)Img.at<uchar>(i, j);
      }
    }
    return w;
  }
}

/**
  * @brief  ������׼������������
  *
  * @param  Path: ��׼ͼƬ·��
  * @param  method: ��ȡ��������ʹ�õķ���
  *   @arg  FeatureSVD: SVD
  *   @arg  FeatureVec: ֱ����ֱΪ����
  *
  * @retval None
  */
void LPR::Standard(std::string Path, FeatureMethod method)
{
  CurrentFeatureMethod = method;
  PlateImg ImgProvider = PlateImg(Path);

  vector<Mat> characters;

  #pragma omp parallel for 
  for (int i = 0; i < ImgProvider.ImgNum; i++)
  {
    cout << "Computing  " << ImgProvider.GetImgName(i) << endl;

    Mat img = imread(ImgProvider.GetImgPath(i));

    binary(img);
    characters.clear();
    Extract(img, characters);
    Mat feature = Feature(characters[0]);

    #pragma omp critical
    {
      StdName.push_back(ImgProvider.GetImgName(i));
      StdFeature.push_back(feature);
    }
  }
}

/**
  * @brief  ʶ��һ���ַ�
  *
  * @param  Img: ֮ǰ�ָ�õĹ�һ������ַ�ͼƬ
  *
  * @retval std::string ʶ����
  *
  */
std::string LPR::IdentifyChar(cv::Mat Img)
{
  Mat feature = Feature(Img);

  switch (CurrentIdentifyMethod)
  {
    /* ���ڷ���ŷ�Ͼ��룩 */
  case IdentifyNeighbor:
    double minDis = norm(feature, StdFeature[0], CV_L2), Dis;
    int minIndex = 0;

    #ifndef _DEBUG
    cout << "--------------------------" << endl;
    cout << StdName[minIndex] << " : " << minDis << endl;
    #endif

    for (int i = 1; i < StdFeature.size(); i++)
    {
      Dis = norm(feature, StdFeature[i], CV_L2);

      if (Dis < minDis)
      {
        minDis = Dis;
        minIndex = i;
      }

      #ifndef _DEBUG
      cout << StdName[i] << " : " << Dis << endl;
      #endif
    }

    #ifndef _DEBUG
    cout << "Neighbor : "<< StdName[minIndex] << " = " << minDis << endl;
    cout << "--------------------------" << endl;
    imshow("LPR", Img);
    waitKey();
    #endif
    return StdName[minIndex];

  }
}


/**
  * @brief  ʶ��һ������ͼ��
  *
  * @param  Img:    ��ʶ��ĳ���ͼ��
  * @param  method: ģʽʶ��ʹ�õķ���
  *   @arg  IdentifyNeighbor:     ���ڷ���ŷ�Ͼ��룩
  *
  * @retval std::string
  */
std::string LPR::Identify(cv::Mat Img, IdentifyMethod method)
{
  CurrentIdentifyMethod = method;

  vector<Mat> characters;
  vector<Mat>::iterator IteImg;

  #ifndef _DEBUG
  imshow("LPR", Img);
  waitKey();
  #endif

  binary(Img);

  #ifndef _DEBUG
  imshow("LPR", Img);
  waitKey();
  #endif

  Extract(Img, characters);

  string result = "";
  for (IteImg = characters.begin(); IteImg != characters.end(); IteImg++)
  {
    result += IdentifyChar(*IteImg);
  }

  return result;
}

