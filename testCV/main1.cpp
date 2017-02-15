/**
  **************************************************************
  * @file       main.cpp
  * @author     高明飞
  * @version    V0.3.1
  * @date       2016-01-04
  *
  * @brief      车牌模式识别主程序入口
  *
  * @details 
  * @verbatim
  * 主程序入口
  *
  * 修改记录：
  * 2015-12-8 :
  *   - File Created.
  *
  * @endverbatim
  ***************************************************************
  */

#include "PlateImg.h"
#include "LPR.h"
#include "locator.h"
#include <windows.h>

using namespace cv;
using namespace std;

/**
  * @brief  车牌图像根目录，最后不要加"\\"
  */
#define IMG_PATH ("./Image")
/**
  * @brief  标准字符图像根目录，最后不要加"\\"
  */
#define STANDARD_PATH ("./Standard")


/**
  * @brief  程序主入口
  *
  * @param  None
  *
  * @retval None
  */
int main1()
{
	Locator locator = Locator("car.png");

	vector<Mat> rs_vec = locator.colorDetect();
	int i=0;
	for(vector<Mat>::iterator it  = rs_vec.begin(); it != rs_vec.end(); ++it){
		
		Mat rs = *(it);

		LPR cur = LPR();
		String result = cur.Identify(rs, IdentifyNeighbor);

		cout << "==========================" << endl;
		cout << "LPR Result: "<< result << endl;
		cout << "==========================" << endl;

		

		waitKey(1000);
	}

	//locator.shapeDetect();


  Mat img;
  string result;
  PlateImg ImgProvider = PlateImg(IMG_PATH);
  LPR curLPR = LPR();

  namedWindow("LPR", CV_WINDOW_NORMAL);

  curLPR.Standard(STANDARD_PATH, FeatureVec);

  for (int i = 0; i < ImgProvider.ImgNum; i++)
  {
    img = imread(ImgProvider.GetImgPath(i));
    result = curLPR.Identify(img, IdentifyNeighbor);

    cout << "==========================" << endl;
    cout << "LPR Result: "<< result << endl;
    cout << "==========================" << endl;
	
    imshow("LPR", img);
    waitKey(1000);
  }


  system("pause");

  return 0;
}



