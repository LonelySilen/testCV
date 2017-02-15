#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "colorDetect.h"
using namespace std;
using namespace cv;
/*********************图像的遍历**********************************************************************/
//第一种方式at<typename>(i,j)
void colorReduce(Mat& image,int div)
{
	for(int i=0;i<image.rows;i++)
	{
		for(int j=0;j<image.cols;j++)
		{
			//取出彩色图像中i行j列第k通道的颜色点
			image.at<Vec3b>(i,j)[0]=image.at<Vec3b>(i,j)[0]/div*div+div/2;
			image.at<Vec3b>(i,j)[1]=image.at<Vec3b>(i,j)[1]/div*div+div/2;
            image.at<Vec3b>(i,j)[2]=image.at<Vec3b>(i,j)[2]/div*div+div/2;
        }
     }
 }
//第二种方式 指针方式
void colorReduce0(const Mat& image,Mat& outImage,int div)
{
	// 创建与原图像等尺寸的图像
    outImage.create(image.size(),image.type());
    int nr=image.rows;
    // 将3通道转换为1通道
    int nl=image.cols*image.channels();
	for(int k=0;k<nr;k++)
    {
		// 每一行图像的指针
		const uchar* inData=image.ptr<uchar>(k);
		uchar* outData=outImage.ptr<uchar>(k);
        for(int i=0;i<nl;i++)
		{
			outData[i]=inData[i]/div*div+div/2;
        }
   }
}
//第三种方式 用迭代器来遍历
void colorReduce1(const Mat& image,Mat& outImage,int div)
{
	outImage.create(image.size(),image.type());
	MatConstIterator_<Vec3b> it_in=image.begin<Vec3b>();
	MatConstIterator_<Vec3b> itend_in=image.end<Vec3b>();
	MatIterator_<Vec3b> it_out=outImage.begin<Vec3b>();
	MatIterator_<Vec3b> itend_out=outImage.end<Vec3b>();
	while(it_in!=itend_in)
	{
		(*it_out)[0]=(*it_in)[0]/div*div+div/2;
		(*it_out)[1]=(*it_in)[1]/div*div+div/2;
		(*it_out)[2]=(*it_in)[2]/div*div+div/2;
		it_in++;
		it_out++;
	}
}
//更简单，更高效
void colorReduce2(const Mat& image,Mat& outImage,int div)
{
	//Mat类把很多算数操作符都进行了重载,让它们来符合矩阵的一些运算,如果+、-、点乘等.
	//下面我们来看看用位操作和基本算术运算来完成本文中的colorReduce程序,它更简单,更高效.
	//将256种灰度阶降到64位其实是抛弃了二进制最后面的4位,所以我们可以用位操作来做这一步处理.
	//首先我们计算2^8降到2^n中的n,然后可以得到mask
	int n=static_cast<int>(log(static_cast<double>(div))/log(2.0));
	uchar mask=0xFF<<n;
	outImage=(image&Scalar(mask,mask,mask))+Scalar(div/2,div/2,div/2);
}
//高效的方式
void colorReduce(const Mat& image,Mat& outImage,int div)
{
	int nr=image.rows;
	int nc=image.cols;
	outImage.create(image.size(),image.type());
	if(image.isContinuous()&&outImage.isContinuous())//检测图像是否连续的函数
	{
		 nr=1;
		 nc=nc*image.rows*image.channels();
	}
	for(int i=0;i<nr;i++)
	{
		const uchar* inData=image.ptr<uchar>(i);
		uchar* outData=outImage.ptr<uchar>(i);
		for(int j=0;j<nc;j++)
		{
			*outData++=*inData++/div*div+div/2;
		}
	}
}

/*********************图像的邻域操作**********************************************************************/
//简单的滤波操作
void ImgFilter2d(const Mat &image,Mat& result)
{
	result.create(image.size(),image.type());
	int nr=image.rows;
	int nc=image.cols*image.channels();
	for(int i=1;i<nr-1;i++)
	{
		const uchar* up_line=image.ptr<uchar>(i-1);//指向上一行
		const uchar* mid_line=image.ptr<uchar>(i);//当前行
		const uchar* down_line=image.ptr<uchar>(i+1);//下一行
		uchar* cur_line=result.ptr<uchar>(i);
		for(int j=1;j<nc-1;j++)
		{
			cur_line[j]=saturate_cast<uchar>(5*mid_line[j]-mid_line[j-1]-mid_line[j+1]-
			 up_line[j]-down_line[j]);
		}
	}
	//1,staturate_cast<typename>是一个类型转换函数,程序里是为了确保运算结果还在uchar范围内
	//2,row和col方法返回图像中的某些行或列,返回值是一个Mat
	//3,setTo方法将Mat对像中的点设置为一个值,Scalar(n)为一个灰度值,Scalar(a,b,c)为一个彩色值

	// 把图像边缘像素设置为0
	result.row(0).setTo(Scalar(0));
	result.row(result.rows-1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols-1).setTo(Scalar(0));
}
/*********************直方图**********************************************************************/
//int nimages：要计算直方图的图像的个数.此函数可以为多图像求直方图,我们通常情况下都只作用于单一图像,所以通常nimages=1.
//const int* channels:图像的通道,它是一个数组,如果是灰度图像则channels[1]={0};如果是彩色图像则channels[3]={0,1,2};如果是只是求彩色图像第2个通道的直方图,则channels[1]={1};
//IuputArray mask:是一个遮罩图像用于确定哪些点参与计算，实际应用中是个很好的参数，默认情况我们都设置为一个空图像，即：Mat().
//OutArray hist:计算得到的直方图
//int dims:得到的直方图的维数,灰度图像为1维,彩色图像为3维.
//const int* histSize:直方图横坐标的区间数.如果是10，则它会横坐标分为10份，然后统计每个区间的像素点总和.
//const float** ranges:这是一个二维数组,用来指出每个区间的范围.
//后面两个参数都有默认值,uniform参数表明直方图是否等距,最后一个参数与多图像下直方图的显示与存储有关.

//灰度直方图
void getHistogram1(const Mat &image,MatND & hist)
{
	cv::Mat image2;
	cvtColor(image,image2,CV_BGR2GRAY);
	const int channels[1]={0};
	const int histSize[1]={256}; // Establish the number of bins
	float hranges[] = { 0, 256 }; /// Set the ranges ( for B,G,R) )
	const float* ranges[1] = { hranges };
  
	bool uniform = true; 
	bool accumulate = false;
	
	// Compute the histograms:
	calcHist(  &image2, 1, channels, Mat(), hist, 1, histSize, ranges, uniform, accumulate );

	//Mat result;
	//equalizeHist(image2,result);
}

//彩色直方图
void getHistogram2(const Mat &image,MatND & hist)
{
	const int channels[3]={0,1,2};
    const int histSize[3]={256,256,256};
    float hranges[2]={0,255};
    const float* ranges[3]={hranges,hranges,hranges};
  
	bool uniform = true; 
	bool accumulate = false;
  
	// Compute the histograms:
	calcHist(  &image, 1, channels, Mat(), hist, 1, histSize, ranges, uniform, accumulate );
}

//不均匀直方图，我们分别统计0-50，50-80，80-150，150-230，230-255区间的灰度分布
void getHistogram3(const Mat &image,MatND & hist)
{
	cv::Mat image2;
	cvtColor(image,image2,CV_BGR2GRAY);

    const int channels[1]={0};
    int histSize[1]={5};
    float hranges[6]={0,50,80,150,230,255};
    const float* ranges[1]={hranges};
  
	bool uniform = true; 
	bool accumulate = false;
  
	// Compute the histograms:
	calcHist(  &image2, 1, channels, Mat(), hist, 1, histSize, ranges, uniform, accumulate );
}
//显示直方图
Mat getHistImg(const MatND& hist)
{
    double maxVal=0;
    double minVal=0;

    //找到直方图中的最大值和最小值
    minMaxLoc(hist,&minVal,&maxVal,0,0);
    int histSize=hist.rows;
    Mat histImg(histSize,histSize,CV_8U,Scalar(255));
    // 设置最大峰值为图像高度的90%
    int hpt=static_cast<int>(0.9*histSize);

    for(int h=0;h<histSize;h++)
    {
        float binVal=hist.at<float>(h);
        int intensity=static_cast<int>(binVal*hpt/maxVal);
        line(histImg,Point(h,histSize),Point(h,histSize-intensity),Scalar::all(0));
    }

    return histImg;
}

//直方图变换
void hisTrans(const Mat &image,MatND & hist,Mat& result,int minValue)
{
	int imax,imin;

	for(imin=0;imin<256;imin++)
	{
		if(hist.at<float>(imin)>minValue)
        break;
	}
	for(imax=255;imax>-1;imax--)
	{
		if(hist.at<float>(imax)>minValue)
        break;
	}
	// 创建一个1*256的矢量
	Mat lut(1,256,CV_8U);
	for(int i=0;i<256;i++)
	{
		if(lut.at<uchar>(i)<imin)
			lut.at<uchar>(i)=0;
		else if(lut.at<uchar>(i)>imax)
			lut.at<uchar>(i)=255;
		else
        lut.at<uchar>(i)=static_cast<uchar>(255.0*(i-imin)/(imax-imin)+0.5);
	}
	LUT(image,lut,result);
}

//直方图的反向映射
Mat reverseMap( Mat &face,Mat &ImgSrc)
{
	//图像降维
    colorReduce(face,face,32);// 样本
    colorReduce(ImgSrc,ImgSrc,32);// 待检测的图像
    
    // 计算颜色直方图
    const int channels[3]={0,1,2};
    const int histSize[3]={256,256,256};
    float hranges[2]={0,255};
    const float* ranges[3]={hranges,hranges,hranges};
	bool uniform = true; 
	bool accumulate = false;
    MatND hist;
    calcHist(	&face, 1, channels, Mat(), hist, 1, histSize, ranges, uniform, accumulate );
		//const int channels[1]={0};
		//int histSize[1]={5};
		//float hranges[6]={0,50,80,150,230,255};
		//const float* ranges[1]={hranges};
	//calcHist(  &image2, 1, channels, Mat(), hist, 1, histSize, ranges, uniform, accumulate );
    // 直方图归一化
    normalize(hist,hist,1.0);

    // 直方图反向映射
    Mat result;
    calcBackProject(&ImgSrc,1,channels,hist,result,ranges,255);
    // 将结果进行阈值化
    threshold(result,result,255*(0.05),255,THRESH_BINARY);
	return result;
}

//
double compareSimilar( Mat &refImg,Mat &image1)
{
    //图像颜色降维
    colorReduce(refImg,64);
    colorReduce(image1,64);

    MatND refH;
	getHistogram2(refImg,refH);
    MatND hist1;
	getHistogram2(image1,hist1);

    double dist1;
    dist1=compareHist(refH,hist1,CV_COMP_BHATTACHARYYA);
	return dist1;
}

/*********************形态学操作**********************************************************************/
//腐蚀运算
Mat erodedImg( Mat &image)
{
	//图像腐蚀
	Mat gray;
    // 彩色转灰度
    cvtColor(image,gray,CV_BGR2GRAY);
    // 阈值化
    threshold(gray,gray,255*(0.5),255,THRESH_BINARY);
    
    // 形态学操作
    // 如果把结构元素设置为Mat()，则将用默认的3*3的矩形结构元素
    Mat eroded;
    erode(gray,eroded,Mat());

	return eroded;
}
//膨胀运算
Mat dilateImg( Mat &image)
{
	//图像腐蚀
	Mat gray ;
    // 彩色转灰度
    cvtColor(image,gray ,CV_BGR2GRAY);
    // 阈值化
    threshold(gray ,gray ,255*(0.5),255,THRESH_BINARY);
    
    // 形态学操作
    // 如果把结构元素设置为Mat()，则将用默认的3*3的矩形结构元素

    Mat dilated;
    dilate(gray ,dilated,Mat());
	return dilated;
}
//对图像进行开运算与闭运算
Mat morphologyExImg( Mat &image, int is_open=0)
{
	//图像腐蚀
	Mat gray ;
    // 彩色转灰度
    cvtColor(image,gray ,CV_BGR2GRAY);
    // 阈值化
    threshold(gray ,gray ,255*(0.5),255,THRESH_BINARY);

    // 定义结构元素
    Mat se(5,5,CV_8U,Scalar(1));
    Mat result;
	if(is_open == 0){
		morphologyEx(gray,result,MORPH_CLOSE,se);
	}else{
		morphologyEx(gray,result,MORPH_OPEN,se);
	}
	return result;
}
//检测边缘,它是对图像先做了一个腐蚀，再做了一次膨胀，然后将两次的结果相减即可
Mat morphGradientImg( Mat &image)
{
	//图像腐蚀
	Mat edge ;
    // 彩色转灰度
    cvtColor(image,edge ,CV_BGR2GRAY);

	morphologyEx(edge,edge,MORPH_GRADIENT,Mat());

    // 阈值化
    threshold(edge,edge,40,255,THRESH_BINARY);

	return edge;
}
//检测角点
//第一个为一个十字型的结构元素,第二个为菱形,第三个是矩形,第四个是一个“X”
//型.然后我们按下面的顺序对一幅图像进行操作,并对最后的结果进行阈值化
Mat cornerDetect( Mat &image)
{
	Mat gray ;
    // 彩色转灰度
    cvtColor(image,gray ,CV_BGR2GRAY);
	// 定义结构元素
    Mat cross(5,5,CV_8U,Scalar(0));
    Mat diamond(5,5,CV_8U,Scalar(1));
    Mat square(5,5,CV_8U,Scalar(1));
    Mat x(5,5,CV_8U,Scalar(0));
    
    for(int i=0;i<5;i++)
    {
        cross.at<uchar>(2,i)=1;
        cross.at<uchar>(i,2)=1;

    }
    diamond.at<uchar>(0,0)=0;
    diamond.at<uchar>(0,1)=0;
    diamond.at<uchar>(1,0)=0;
    diamond.at<uchar>(4,4)=0;
    diamond.at<uchar>(3,4)=0;
    diamond.at<uchar>(4,3)=0;
    diamond.at<uchar>(4,0)=0;
    diamond.at<uchar>(4,1)=0;
    diamond.at<uchar>(3,0)=0;
    diamond.at<uchar>(0,4)=0;
    diamond.at<uchar>(0,3)=0;
    diamond.at<uchar>(1,4)=0;

    for(int i=0;i<5;i++){
        x.at<uchar>(i,i)=1;
        x.at<uchar>(4-i,i)=1;
    }

	Mat result;
	dilate(gray,result,cross);
	erode(result,result,diamond);

	Mat result2;
	dilate(gray,result2,x);
	erode(result2,result2,square);
	absdiff(result2,result,result);

	threshold(result,result,40,255,THRESH_BINARY);
	return result;
}
// 标记角点
void drawOnImage(const Mat& binary,Mat& image)
{
    for(int i=0;i<binary.rows;i++)
    {
        // 获取行指针
        const uchar* data=binary.ptr<uchar>(i);
        for(int j=0;j<binary.cols;j++)
        {
            if(data[j]) //角点图像上的白点
                circle(image,Point(j,i),8,Scalar(0,255,0));// 画圈
        }
    }
}
/*********************滤波器**********************************************************************/

//三种滤波器
Mat medBlur( Mat &image)
{

	Mat result;
    cvtColor(image,result,CV_BGR2GRAY);

    medianBlur(result,result,5);

	return result;
}
Mat simpleBlur( Mat &image)
{

	Mat result;
    cvtColor(image,result,CV_BGR2GRAY);

    blur(result,result,Size(5,5));

	return result;
}
Mat gaussiBlur( Mat &image)
{

	Mat result;
    cvtColor(image,result,CV_BGR2GRAY);

    GaussianBlur(result,result,Size(5,5),1.5);

	return result;
}
//Sobel边缘检测在高通滤波的实例
#define SOBEL_HORZ 0
#define SOBEL_VERT 1
#define SOBEL_BOTH 2
bool sobel(const Mat& image,Mat& result,int TYPE)
{
	Mat gray ;
    // 彩色转灰度
    cvtColor(image,gray ,CV_BGR2GRAY);

	if(gray.channels()!=1)
        return false;
    // 系数设置
    int kx(0);
    int ky(0);
    if( TYPE == SOBEL_HORZ ){
        kx=0;ky=1;
    }
    else if( TYPE == SOBEL_VERT ){
        kx=1;ky=0;
    }
    else if( TYPE == SOBEL_BOTH ){
        kx=1;ky=1;
    }
    else
        return false;

    // 设置mask
    float mask[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
    Mat y_mask=Mat(3,3,CV_32F,mask)/8;
    Mat x_mask=y_mask.t(); // 转置

    // 计算x方向和y方向上的滤波
    Mat sobelX,sobelY;
    filter2D(gray,sobelX,CV_32F,x_mask);
    filter2D(gray,sobelY,CV_32F,y_mask);
    sobelX=abs(sobelX);
    sobelY=abs(sobelY);
    // 梯度图
    Mat gradient=kx*sobelX.mul(sobelX)+ky*sobelY.mul(sobelY);

    // 计算阈值
    int scale=4;
    double cutoff=scale*mean(gradient)[0];

    result.create(gray.size(),gray.type());
    result.setTo(0);
    for(int i=1;i<gray.rows-1;i++)
    {
        float* sbxPtr=sobelX.ptr<float>(i);
        float* sbyPtr=sobelY.ptr<float>(i);
        float* prePtr=gradient.ptr<float>(i-1);
        float* curPtr=gradient.ptr<float>(i);
        float* lstPtr=gradient.ptr<float>(i+1);
        uchar* rstPtr=result.ptr<uchar>(i);
        // 阈值化和极大值抑制
        for(int j=1;j<gray.cols-1;j++)
        {
            if( curPtr[j]>cutoff && (
                (sbxPtr[j]>kx*sbyPtr[j] && curPtr[j]>curPtr[j-1] && curPtr[j]>curPtr[j+1]) ||
                (sbyPtr[j]>ky*sbxPtr[j] && curPtr[j]>prePtr[j] && curPtr[j]>lstPtr[j]) ))
                rstPtr[j]=255;
        }
    }

    return true;
}
//Canny边缘检测
Mat cannyDetect( Mat &image)
{
	Mat gray;
    cvtColor(image,gray,CV_BGR2GRAY);

	Mat contours;
	Canny(gray,contours,125,350);
	threshold(contours,contours,128,255,THRESH_BINARY);

	return contours;
}
//Hairrs角点检测
Mat hairrsDetect( Mat &image)
{
    Mat gray;
    cvtColor(image,gray,CV_BGR2GRAY);

	Mat cornerStrength;
    cornerHarris(gray,cornerStrength,3,3,0.01);

    double maxStrength;
    double minStrength;
    // 找到图像中的最大、最小值
    minMaxLoc(cornerStrength,&minStrength,&maxStrength);

    Mat dilated;
    Mat locaMax;
    // 膨胀图像，最找出图像中全部的局部最大值点
    dilate(cornerStrength,dilated,Mat());
    // compare是一个逻辑比较函数，返回两幅图像中对应点相同的二值图像
    compare(cornerStrength,dilated,locaMax,CMP_EQ);

    Mat cornerMap;
    double qualityLevel=0.01;
    double th=qualityLevel*maxStrength; // 阈值计算
    threshold(cornerStrength,cornerMap,th,255,THRESH_BINARY);
    cornerMap.convertTo(cornerMap,CV_8U);
    // 逐点的位运算
    bitwise_and(cornerMap,locaMax,cornerMap);

	return cornerMap;
}
//画出Hairrs角点
void drawCornerOnImage(Mat& image,const Mat&binary)
{
    Mat_<uchar>::const_iterator it=binary.begin<uchar>();
    Mat_<uchar>::const_iterator itd=binary.end<uchar>();
    for(int i=0;it!=itd;it++,i++)
    {
        if(*it)
            circle(image,Point(i%image.cols,i/image.cols),3,Scalar(0,255,0),1);    
    }
}
//HoughLinesP检测直线的例子
vector<Vec4i> houghLinesDetect( Mat &image)
{
	Mat gray;
    cvtColor(image,gray,CV_BGR2GRAY);

	Mat contours;
	Canny(gray,contours,125,350);
	threshold(contours,contours,128,255,THRESH_BINARY);

	vector<Vec4i> lines;
	// 检测直线，最小投票为90，线条不短于50，间隙不小于10
	HoughLinesP(contours,lines,1,CV_PI/180,80,50,10);
	return lines;
}
//将检测到的直线在图上画出来
void drawDetectLines(Mat& image,const vector<Vec4i>& lines,Scalar & color)
{
	// 将检测到的直线在图上画出来
	vector<Vec4i>::const_iterator it=lines.begin();
	while(it!=lines.end())
	{
		Point pt1((*it)[0],(*it)[1]);
		Point pt2((*it)[2],(*it)[3]);
		line(image,pt1,pt2,color,2); //  线条宽度设置为2
		++it;
	}
}

void myDrawContours(Mat& result,vector<vector<Point>>& contours)
{
	// 轮廓表示为一个矩形
	Rect r = boundingRect(Mat(contours[0]));
	rectangle(result, r, Scalar(255), 2);

	// 轮廓表示为一个圆
	float radius;
	Point2f center;
	minEnclosingCircle(Mat(contours[1]), center, radius);
	circle(result, Point(center), static_cast<int>(radius), Scalar(255), 2);

	// 轮廓表示为一个多边形
	vector<Point> poly;
	approxPolyDP(Mat(contours[2]), poly, 5, true);
	vector<Point>::const_iterator itp = poly.begin();
	while (itp != (poly.end() - 1))
	{
		line(result, *itp, *(itp + 1), Scalar(255), 2);
		++itp;
	}
	line(result, *itp, *(poly.begin()), Scalar(255), 2);
	// 轮廓表示为凸多边形
	vector<Point> hull;
	convexHull(Mat(contours[3]), hull);
	vector<Point>::const_iterator ith = hull.begin();
	while (ith != (hull.end() - 1))
	{
		line(result, *ith, *(ith + 1), Scalar(255), 2);
		++ith;
	}
	line(result, *ith, *(hull.begin()), Scalar(255), 2);
}
/*********************主函数**********************************************************************/
//int main()
//{
	/*// 读入一张图片
	cv::Mat image;
	cv::Mat image1(320,320,CV_8U,cv::Scalar(100));
	image=cv::imread("cat.jpg");

	cv::namedWindow("cat");
	cv::imshow("cat",image);
	cv::waitKey(2000);
	Mat newImage;*/
	//newImage=image.clone();
	//colorReduce(newImage, 64);
	//colorReduce(image, newImage, 64);
	//ImgFilter2d(image, newImage);

	//colorDetect cdetect;
	//cdetect.SetTargetColor(150,150,150); 
	//cdetect.SetMinDistance(50);
	//newImage = cdetect.process(image);
	
	//imwrite("output.jpg",newImage);
	//return 0;

	//直方图变换
	/*cv::Mat image;
	Mat newImage;
	image=cv::imread("cat.jpg");
	MatND  hist;
	getHistogram1(image, hist);

	//newImage = getHistImg(hist);
	hisTrans(image,hist,newImage,100);
	imwrite("output.jpg",newImage);
    return 0;*/

	//反向映射
	//Mat face=imread("face.png");        // 样本
    //Mat ImgSrc=imread("img.png");    // 待检测的图像
	//Mat result = reverseMap(face,ImgSrc );
	//imwrite("output.jpg",result);
	//return 0;

	//相似性
	/*Mat refImg=imread("face.png");
    Mat image1=imread("girl.jpg");
    Mat image2=imread("img.png");

    double dist1,dist2;
    dist1=compareSimilar(refImg,image1);
    dist2=compareSimilar(refImg,image2);

    std::cout<<"dist1="<<dist1<<",dist2="<<dist2<<std::endl;*/


	//图像腐蚀
	//cv::Mat image;
	//Mat newImage;
	//image=cv::imread("cat.jpg");
	//newImage = erodedImg(image);   
	//newImage = dilateImg(image);
	//imwrite("output.jpg",newImage);
    //return 0;

	/*//开运算与闭运算
	Mat image=cv::imread("cat.jpg");
	Mat newImage;
	newImage = morphologyExImg(image);
    //newImage = morphologyExImg(image,1);
	imwrite("output.jpg",newImage);
    return 0;*/

	//边缘检测
	//Mat image=cv::imread("cat.jpg");
	//Mat newImage;
	//newImage = morphGradientImg(image);
	//imwrite("output.jpg",newImage);
    //return 0;

	/*//角点检测
	Mat image=cv::imread("cat.jpg");
	Mat newImage;
	newImage = cornerDetect(image);
	// 标记角点
	drawOnImage(newImage, image);
	imwrite("output.jpg",image);
    return 0;*/
	
	/*//车牌定位
	Mat cimage=imread("car.jpg");
    Mat image;
    cvtColor(cimage,image,CV_BGR2GRAY); 
    Mat result;
    //检测竖直边缘
    morphologyEx(image,result,MORPH_GRADIENT,Mat(1,2,CV_8U,Scalar(1)));
    //阈值化
    threshold(result,result,255*(0.2),255,THRESH_BINARY);
    //水平方向闭运算
    morphologyEx(result,result,MORPH_CLOSE,Mat(1,20,CV_8U,Scalar(1)));
    //竖起方向闭运算
    morphologyEx(result,result,MORPH_CLOSE,Mat(10,1,CV_8U,Scalar(1)));
	imwrite("output.jpg",result);
    return 0;*/

	/*//三种滤波器的效果
	Mat image=cv::imread("cat.jpg");
	Mat newImage;

    Mat blurResult;
    Mat gaussianResult;
    Mat medianResult;

    blurResult = simpleBlur(image);
	gaussianResult = gaussiBlur(image);
	medianResult = medBlur(image);

    namedWindow("blur");imshow("blur",blurResult);
	namedWindow("Gaussianblur");imshow("Gaussianblur",gaussianResult);
	namedWindow("medianBlur");imshow("medianBlur",medianResult);

    waitKey();
    return 0;*/


	/*//Sobel轮廓检测在高通滤波的实例
	Mat image=cv::imread("cat.jpg");
	Mat newImage;
	if(sobel(image, newImage, 2)){
		imwrite("output.jpg",newImage);
	}
    return 0;*/

	/*//Canny边缘检测在高通滤波的实例
	Mat image=cv::imread("cat.jpg");
	Mat newImage;
	newImage = cannyDetect(image);
	imwrite("output.jpg",newImage);
    return 0;*/

	/*//HoughLinesP检测直线的例子
	Mat image=cv::imread("car.jpg");

	vector<Vec4i> lines;
	lines = houghLinesDetect(image);
	drawDetectLines(image,lines,Scalar(0,255,0));

	namedWindow("Lines");
	imshow("Lines",image);
	waitKey();
	return 0;*/

	/*//轮廓的提取与描述
	Mat image=imread("team.png");
	Mat newImage;
	cvtColor(image,newImage,CV_BGR2GRAY);
	vector<vector<Point>> contours;
	// find
	findContours(newImage,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
	// draw
	Mat result(newImage.size(),CV_8U,Scalar(0));
	//drawContours(result,contours,-1,Scalar(255),2);
	myDrawContours(result,contours);

	namedWindow("contours");
	imshow("contours",result);
	waitKey();
	return 0;*/

	/*//Hairrs角点检测
	Mat image=imread("cat.jpg");

    Mat cornerMap = hairrsDetect(image);
	drawCornerOnImage(image,cornerMap);
	namedWindow("cornerStrength");
	imshow("cornerStrength",image);
	waitKey();
    return 0;*/


    
//    return 0;

//}


