/**
  **************************************************************
  * @file       LPR.h
  * @author     ������
  * @version    V0.3.1
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

#pragma once

#include <iostream>
#include <opencv2\opencv.hpp>
#include <vector>

/**
  * @brief  �ַ�ͼƬ��һ�����ͼ����
  */
#define CharImgWidth 60
/**
  * @brief  �ַ�ͼƬ��һ�����ͼ��߶�
  */
#define CharImgHeight 100

/**
  * @brief  ��ȡ������������ʹ�õķ���
  */
typedef enum
{
  FeatureSVD,         /*!< SVD���� */
  FeatureVec,         /*!< ֱ����ֱΪ���� */
} FeatureMethod;

/**
  * @brief  ����ģʽʶ�����ʹ�õķ���
  */
typedef enum
{
  IdentifyNeighbor,     /*!< ���ڷ���ŷ�Ͼ��룩 */

} IdentifyMethod;

/**
  * @brief  �������ڽ��г���ʶ��
  */
class LPR
{
public:
  LPR();

  void Standard(std::string Path, FeatureMethod method);
  std::string Identify(cv::Mat Img, IdentifyMethod method);

  void binary(cv::Mat & Img);
  void Extract(cv::Mat & Img, std::vector<cv::Mat> & vec);
  cv::Mat Feature(cv::Mat Img);
  std::string IdentifyChar(cv::Mat Img);

  ~LPR();
private:
  std::vector<std::string> StdName;     /*!< ��׼�������� */
  std::vector<cv::Mat> StdFeature;      /*!< ��׼������������ */
  FeatureMethod CurrentFeatureMethod;   /*!< ��ǰʹ�õ�����������ȡ���� */
  IdentifyMethod CurrentIdentifyMethod; /*!< ��ǰʹ�õ�ģʽʶ�𷽷� */
};

