/**
  **************************************************************
  * @file       PlateImg.cpp
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

#include "PlateImg.h"
#include<iostream>

/**
  * @brief  ���캯��
  *
  * @param  path: ͼƬ·����·���±������Index.txt�����ļ�
  *
  * @retval None
  */
PlateImg::PlateImg(string path)
{
  ImgNum = 0;
  ImgPath = path;
  path += "/Index.txt";
  ifstream f = ifstream(path);
  string s;
  while (f >> s)
  {
    ImgName.push_back(s);
    ImgNum++;
  }
  f.close();
}

/**
  * @brief  ��ȡͼ������
  *
  * @param  index:  ������
  *
  * @retval string: ͼ�����ƣ�������׺��·����
  */
string PlateImg::GetImgName(int index)
{
  if (index >= ImgNum)
    return "";

  return ImgName[index];
}

/**
* @brief  ��ȡͼ������·��
*
* @param  index:  ������
*
* @retval string: ͼ������·����������׺����׺�̶�Ϊ.png
*/
string PlateImg::GetImgPath(int index)
{
  if (index >= ImgNum)
    return "";

  return (ImgPath + "/" + GetImgName(index) + ".png");
}
