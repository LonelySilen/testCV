/**
  **************************************************************
  * @file       PlateImg.h
  * @author     ������
  * @version    V1.0
  * @date       2015-12-8
  *
  * @brief      �����ṩ����ͼ�����Ƽ�·������
  *
  * @details 
  * @verbatim
  * ���������ļ�ö���ļ��������еĳ���ͼ��
  *
  * �޸ļ�¼��
  * 2015-12-8 :
  *   - File Created.
  *
  * @endverbatim
  ***************************************************************
  */

#pragma once

#include <string>
#include <fstream>
#include <vector>

using namespace std;

/**
  * @brief  �����ṩ����ͼ�����Ƽ�·������
  */
class PlateImg
{
public:
  PlateImg(string path);

  string GetImgName(int index);

  string GetImgPath(int index);

  int ImgNum;           /*!< ͼ������ */

private:
  string ImgPath;           /*!< ͼ���Ŀ¼ */

  vector <string> ImgName;  /*!< ͼ������ */
};

