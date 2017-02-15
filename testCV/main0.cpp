#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "colorDetect.h"
using namespace std;
using namespace cv;
/*********************ͼ��ı���**********************************************************************/
//��һ�ַ�ʽat<typename>(i,j)
void colorReduce(Mat& image,int div)
{
	for(int i=0;i<image.rows;i++)
	{
		for(int j=0;j<image.cols;j++)
		{
			//ȡ����ɫͼ����i��j�е�kͨ������ɫ��
			image.at<Vec3b>(i,j)[0]=image.at<Vec3b>(i,j)[0]/div*div+div/2;
			image.at<Vec3b>(i,j)[1]=image.at<Vec3b>(i,j)[1]/div*div+div/2;
            image.at<Vec3b>(i,j)[2]=image.at<Vec3b>(i,j)[2]/div*div+div/2;
        }
     }
 }
//�ڶ��ַ�ʽ ָ�뷽ʽ
void colorReduce0(const Mat& image,Mat& outImage,int div)
{
	// ������ԭͼ��ȳߴ��ͼ��
    outImage.create(image.size(),image.type());
    int nr=image.rows;
    // ��3ͨ��ת��Ϊ1ͨ��
    int nl=image.cols*image.channels();
	for(int k=0;k<nr;k++)
    {
		// ÿһ��ͼ���ָ��
		const uchar* inData=image.ptr<uchar>(k);
		uchar* outData=outImage.ptr<uchar>(k);
        for(int i=0;i<nl;i++)
		{
			outData[i]=inData[i]/div*div+div/2;
        }
   }
}
//�����ַ�ʽ �õ�����������
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
//���򵥣�����Ч
void colorReduce2(const Mat& image,Mat& outImage,int div)
{
	//Mat��Ѻܶ�����������������������,�����������Ͼ����һЩ����,���+��-����˵�.
	//����������������λ�����ͻ���������������ɱ����е�colorReduce����,������,����Ч.
	//��256�ֻҶȽ׽���64λ��ʵ�������˶�����������4λ,�������ǿ�����λ����������һ������.
	//�������Ǽ���2^8����2^n�е�n,Ȼ����Եõ�mask
	int n=static_cast<int>(log(static_cast<double>(div))/log(2.0));
	uchar mask=0xFF<<n;
	outImage=(image&Scalar(mask,mask,mask))+Scalar(div/2,div/2,div/2);
}
//��Ч�ķ�ʽ
void colorReduce(const Mat& image,Mat& outImage,int div)
{
	int nr=image.rows;
	int nc=image.cols;
	outImage.create(image.size(),image.type());
	if(image.isContinuous()&&outImage.isContinuous())//���ͼ���Ƿ������ĺ���
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

/*********************ͼ����������**********************************************************************/
//�򵥵��˲�����
void ImgFilter2d(const Mat &image,Mat& result)
{
	result.create(image.size(),image.type());
	int nr=image.rows;
	int nc=image.cols*image.channels();
	for(int i=1;i<nr-1;i++)
	{
		const uchar* up_line=image.ptr<uchar>(i-1);//ָ����һ��
		const uchar* mid_line=image.ptr<uchar>(i);//��ǰ��
		const uchar* down_line=image.ptr<uchar>(i+1);//��һ��
		uchar* cur_line=result.ptr<uchar>(i);
		for(int j=1;j<nc-1;j++)
		{
			cur_line[j]=saturate_cast<uchar>(5*mid_line[j]-mid_line[j-1]-mid_line[j+1]-
			 up_line[j]-down_line[j]);
		}
	}
	//1,staturate_cast<typename>��һ������ת������,��������Ϊ��ȷ������������uchar��Χ��
	//2,row��col��������ͼ���е�ĳЩ�л���,����ֵ��һ��Mat
	//3,setTo������Mat�����еĵ�����Ϊһ��ֵ,Scalar(n)Ϊһ���Ҷ�ֵ,Scalar(a,b,c)Ϊһ����ɫֵ

	// ��ͼ���Ե��������Ϊ0
	result.row(0).setTo(Scalar(0));
	result.row(result.rows-1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols-1).setTo(Scalar(0));
}
/*********************ֱ��ͼ**********************************************************************/
//int nimages��Ҫ����ֱ��ͼ��ͼ��ĸ���.�˺�������Ϊ��ͼ����ֱ��ͼ,����ͨ������¶�ֻ�����ڵ�һͼ��,����ͨ��nimages=1.
//const int* channels:ͼ���ͨ��,����һ������,����ǻҶ�ͼ����channels[1]={0};����ǲ�ɫͼ����channels[3]={0,1,2};�����ֻ�����ɫͼ���2��ͨ����ֱ��ͼ,��channels[1]={1};
//IuputArray mask:��һ������ͼ������ȷ����Щ�������㣬ʵ��Ӧ�����Ǹ��ܺõĲ�����Ĭ��������Ƕ�����Ϊһ����ͼ�񣬼���Mat().
//OutArray hist:����õ���ֱ��ͼ
//int dims:�õ���ֱ��ͼ��ά��,�Ҷ�ͼ��Ϊ1ά,��ɫͼ��Ϊ3ά.
//const int* histSize:ֱ��ͼ�������������.�����10��������������Ϊ10�ݣ�Ȼ��ͳ��ÿ����������ص��ܺ�.
//const float** ranges:����һ����ά����,����ָ��ÿ������ķ�Χ.
//����������������Ĭ��ֵ,uniform��������ֱ��ͼ�Ƿ�Ⱦ�,���һ���������ͼ����ֱ��ͼ����ʾ��洢�й�.

//�Ҷ�ֱ��ͼ
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

//��ɫֱ��ͼ
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

//������ֱ��ͼ�����Ƿֱ�ͳ��0-50��50-80��80-150��150-230��230-255����ĻҶȷֲ�
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
//��ʾֱ��ͼ
Mat getHistImg(const MatND& hist)
{
    double maxVal=0;
    double minVal=0;

    //�ҵ�ֱ��ͼ�е����ֵ����Сֵ
    minMaxLoc(hist,&minVal,&maxVal,0,0);
    int histSize=hist.rows;
    Mat histImg(histSize,histSize,CV_8U,Scalar(255));
    // ��������ֵΪͼ��߶ȵ�90%
    int hpt=static_cast<int>(0.9*histSize);

    for(int h=0;h<histSize;h++)
    {
        float binVal=hist.at<float>(h);
        int intensity=static_cast<int>(binVal*hpt/maxVal);
        line(histImg,Point(h,histSize),Point(h,histSize-intensity),Scalar::all(0));
    }

    return histImg;
}

//ֱ��ͼ�任
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
	// ����һ��1*256��ʸ��
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

//ֱ��ͼ�ķ���ӳ��
Mat reverseMap( Mat &face,Mat &ImgSrc)
{
	//ͼ��ά
    colorReduce(face,face,32);// ����
    colorReduce(ImgSrc,ImgSrc,32);// ������ͼ��
    
    // ������ɫֱ��ͼ
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
    // ֱ��ͼ��һ��
    normalize(hist,hist,1.0);

    // ֱ��ͼ����ӳ��
    Mat result;
    calcBackProject(&ImgSrc,1,channels,hist,result,ranges,255);
    // �����������ֵ��
    threshold(result,result,255*(0.05),255,THRESH_BINARY);
	return result;
}

//
double compareSimilar( Mat &refImg,Mat &image1)
{
    //ͼ����ɫ��ά
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

/*********************��̬ѧ����**********************************************************************/
//��ʴ����
Mat erodedImg( Mat &image)
{
	//ͼ��ʴ
	Mat gray;
    // ��ɫת�Ҷ�
    cvtColor(image,gray,CV_BGR2GRAY);
    // ��ֵ��
    threshold(gray,gray,255*(0.5),255,THRESH_BINARY);
    
    // ��̬ѧ����
    // ����ѽṹԪ������ΪMat()������Ĭ�ϵ�3*3�ľ��νṹԪ��
    Mat eroded;
    erode(gray,eroded,Mat());

	return eroded;
}
//��������
Mat dilateImg( Mat &image)
{
	//ͼ��ʴ
	Mat gray ;
    // ��ɫת�Ҷ�
    cvtColor(image,gray ,CV_BGR2GRAY);
    // ��ֵ��
    threshold(gray ,gray ,255*(0.5),255,THRESH_BINARY);
    
    // ��̬ѧ����
    // ����ѽṹԪ������ΪMat()������Ĭ�ϵ�3*3�ľ��νṹԪ��

    Mat dilated;
    dilate(gray ,dilated,Mat());
	return dilated;
}
//��ͼ����п������������
Mat morphologyExImg( Mat &image, int is_open=0)
{
	//ͼ��ʴ
	Mat gray ;
    // ��ɫת�Ҷ�
    cvtColor(image,gray ,CV_BGR2GRAY);
    // ��ֵ��
    threshold(gray ,gray ,255*(0.5),255,THRESH_BINARY);

    // ����ṹԪ��
    Mat se(5,5,CV_8U,Scalar(1));
    Mat result;
	if(is_open == 0){
		morphologyEx(gray,result,MORPH_CLOSE,se);
	}else{
		morphologyEx(gray,result,MORPH_OPEN,se);
	}
	return result;
}
//����Ե,���Ƕ�ͼ��������һ����ʴ��������һ�����ͣ�Ȼ�����εĽ���������
Mat morphGradientImg( Mat &image)
{
	//ͼ��ʴ
	Mat edge ;
    // ��ɫת�Ҷ�
    cvtColor(image,edge ,CV_BGR2GRAY);

	morphologyEx(edge,edge,MORPH_GRADIENT,Mat());

    // ��ֵ��
    threshold(edge,edge,40,255,THRESH_BINARY);

	return edge;
}
//���ǵ�
//��һ��Ϊһ��ʮ���͵ĽṹԪ��,�ڶ���Ϊ����,�������Ǿ���,���ĸ���һ����X��
//��.Ȼ�����ǰ������˳���һ��ͼ����в���,�������Ľ��������ֵ��
Mat cornerDetect( Mat &image)
{
	Mat gray ;
    // ��ɫת�Ҷ�
    cvtColor(image,gray ,CV_BGR2GRAY);
	// ����ṹԪ��
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
// ��ǽǵ�
void drawOnImage(const Mat& binary,Mat& image)
{
    for(int i=0;i<binary.rows;i++)
    {
        // ��ȡ��ָ��
        const uchar* data=binary.ptr<uchar>(i);
        for(int j=0;j<binary.cols;j++)
        {
            if(data[j]) //�ǵ�ͼ���ϵİ׵�
                circle(image,Point(j,i),8,Scalar(0,255,0));// ��Ȧ
        }
    }
}
/*********************�˲���**********************************************************************/

//�����˲���
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
//Sobel��Ե����ڸ�ͨ�˲���ʵ��
#define SOBEL_HORZ 0
#define SOBEL_VERT 1
#define SOBEL_BOTH 2
bool sobel(const Mat& image,Mat& result,int TYPE)
{
	Mat gray ;
    // ��ɫת�Ҷ�
    cvtColor(image,gray ,CV_BGR2GRAY);

	if(gray.channels()!=1)
        return false;
    // ϵ������
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

    // ����mask
    float mask[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
    Mat y_mask=Mat(3,3,CV_32F,mask)/8;
    Mat x_mask=y_mask.t(); // ת��

    // ����x�����y�����ϵ��˲�
    Mat sobelX,sobelY;
    filter2D(gray,sobelX,CV_32F,x_mask);
    filter2D(gray,sobelY,CV_32F,y_mask);
    sobelX=abs(sobelX);
    sobelY=abs(sobelY);
    // �ݶ�ͼ
    Mat gradient=kx*sobelX.mul(sobelX)+ky*sobelY.mul(sobelY);

    // ������ֵ
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
        // ��ֵ���ͼ���ֵ����
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
//Canny��Ե���
Mat cannyDetect( Mat &image)
{
	Mat gray;
    cvtColor(image,gray,CV_BGR2GRAY);

	Mat contours;
	Canny(gray,contours,125,350);
	threshold(contours,contours,128,255,THRESH_BINARY);

	return contours;
}
//Hairrs�ǵ���
Mat hairrsDetect( Mat &image)
{
    Mat gray;
    cvtColor(image,gray,CV_BGR2GRAY);

	Mat cornerStrength;
    cornerHarris(gray,cornerStrength,3,3,0.01);

    double maxStrength;
    double minStrength;
    // �ҵ�ͼ���е������Сֵ
    minMaxLoc(cornerStrength,&minStrength,&maxStrength);

    Mat dilated;
    Mat locaMax;
    // ����ͼ�����ҳ�ͼ����ȫ���ľֲ����ֵ��
    dilate(cornerStrength,dilated,Mat());
    // compare��һ���߼��ȽϺ�������������ͼ���ж�Ӧ����ͬ�Ķ�ֵͼ��
    compare(cornerStrength,dilated,locaMax,CMP_EQ);

    Mat cornerMap;
    double qualityLevel=0.01;
    double th=qualityLevel*maxStrength; // ��ֵ����
    threshold(cornerStrength,cornerMap,th,255,THRESH_BINARY);
    cornerMap.convertTo(cornerMap,CV_8U);
    // ����λ����
    bitwise_and(cornerMap,locaMax,cornerMap);

	return cornerMap;
}
//����Hairrs�ǵ�
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
//HoughLinesP���ֱ�ߵ�����
vector<Vec4i> houghLinesDetect( Mat &image)
{
	Mat gray;
    cvtColor(image,gray,CV_BGR2GRAY);

	Mat contours;
	Canny(gray,contours,125,350);
	threshold(contours,contours,128,255,THRESH_BINARY);

	vector<Vec4i> lines;
	// ���ֱ�ߣ���СͶƱΪ90������������50����϶��С��10
	HoughLinesP(contours,lines,1,CV_PI/180,80,50,10);
	return lines;
}
//����⵽��ֱ����ͼ�ϻ�����
void drawDetectLines(Mat& image,const vector<Vec4i>& lines,Scalar & color)
{
	// ����⵽��ֱ����ͼ�ϻ�����
	vector<Vec4i>::const_iterator it=lines.begin();
	while(it!=lines.end())
	{
		Point pt1((*it)[0],(*it)[1]);
		Point pt2((*it)[2],(*it)[3]);
		line(image,pt1,pt2,color,2); //  �����������Ϊ2
		++it;
	}
}

void myDrawContours(Mat& result,vector<vector<Point>>& contours)
{
	// ������ʾΪһ������
	Rect r = boundingRect(Mat(contours[0]));
	rectangle(result, r, Scalar(255), 2);

	// ������ʾΪһ��Բ
	float radius;
	Point2f center;
	minEnclosingCircle(Mat(contours[1]), center, radius);
	circle(result, Point(center), static_cast<int>(radius), Scalar(255), 2);

	// ������ʾΪһ�������
	vector<Point> poly;
	approxPolyDP(Mat(contours[2]), poly, 5, true);
	vector<Point>::const_iterator itp = poly.begin();
	while (itp != (poly.end() - 1))
	{
		line(result, *itp, *(itp + 1), Scalar(255), 2);
		++itp;
	}
	line(result, *itp, *(poly.begin()), Scalar(255), 2);
	// ������ʾΪ͹�����
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
/*********************������**********************************************************************/
//int main()
//{
	/*// ����һ��ͼƬ
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

	//ֱ��ͼ�任
	/*cv::Mat image;
	Mat newImage;
	image=cv::imread("cat.jpg");
	MatND  hist;
	getHistogram1(image, hist);

	//newImage = getHistImg(hist);
	hisTrans(image,hist,newImage,100);
	imwrite("output.jpg",newImage);
    return 0;*/

	//����ӳ��
	//Mat face=imread("face.png");        // ����
    //Mat ImgSrc=imread("img.png");    // ������ͼ��
	//Mat result = reverseMap(face,ImgSrc );
	//imwrite("output.jpg",result);
	//return 0;

	//������
	/*Mat refImg=imread("face.png");
    Mat image1=imread("girl.jpg");
    Mat image2=imread("img.png");

    double dist1,dist2;
    dist1=compareSimilar(refImg,image1);
    dist2=compareSimilar(refImg,image2);

    std::cout<<"dist1="<<dist1<<",dist2="<<dist2<<std::endl;*/


	//ͼ��ʴ
	//cv::Mat image;
	//Mat newImage;
	//image=cv::imread("cat.jpg");
	//newImage = erodedImg(image);   
	//newImage = dilateImg(image);
	//imwrite("output.jpg",newImage);
    //return 0;

	/*//�������������
	Mat image=cv::imread("cat.jpg");
	Mat newImage;
	newImage = morphologyExImg(image);
    //newImage = morphologyExImg(image,1);
	imwrite("output.jpg",newImage);
    return 0;*/

	//��Ե���
	//Mat image=cv::imread("cat.jpg");
	//Mat newImage;
	//newImage = morphGradientImg(image);
	//imwrite("output.jpg",newImage);
    //return 0;

	/*//�ǵ���
	Mat image=cv::imread("cat.jpg");
	Mat newImage;
	newImage = cornerDetect(image);
	// ��ǽǵ�
	drawOnImage(newImage, image);
	imwrite("output.jpg",image);
    return 0;*/
	
	/*//���ƶ�λ
	Mat cimage=imread("car.jpg");
    Mat image;
    cvtColor(cimage,image,CV_BGR2GRAY); 
    Mat result;
    //�����ֱ��Ե
    morphologyEx(image,result,MORPH_GRADIENT,Mat(1,2,CV_8U,Scalar(1)));
    //��ֵ��
    threshold(result,result,255*(0.2),255,THRESH_BINARY);
    //ˮƽ���������
    morphologyEx(result,result,MORPH_CLOSE,Mat(1,20,CV_8U,Scalar(1)));
    //�����������
    morphologyEx(result,result,MORPH_CLOSE,Mat(10,1,CV_8U,Scalar(1)));
	imwrite("output.jpg",result);
    return 0;*/

	/*//�����˲�����Ч��
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


	/*//Sobel��������ڸ�ͨ�˲���ʵ��
	Mat image=cv::imread("cat.jpg");
	Mat newImage;
	if(sobel(image, newImage, 2)){
		imwrite("output.jpg",newImage);
	}
    return 0;*/

	/*//Canny��Ե����ڸ�ͨ�˲���ʵ��
	Mat image=cv::imread("cat.jpg");
	Mat newImage;
	newImage = cannyDetect(image);
	imwrite("output.jpg",newImage);
    return 0;*/

	/*//HoughLinesP���ֱ�ߵ�����
	Mat image=cv::imread("car.jpg");

	vector<Vec4i> lines;
	lines = houghLinesDetect(image);
	drawDetectLines(image,lines,Scalar(0,255,0));

	namedWindow("Lines");
	imshow("Lines",image);
	waitKey();
	return 0;*/

	/*//��������ȡ������
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

	/*//Hairrs�ǵ���
	Mat image=imread("cat.jpg");

    Mat cornerMap = hairrsDetect(image);
	drawCornerOnImage(image,cornerMap);
	namedWindow("cornerStrength");
	imshow("cornerStrength",image);
	waitKey();
    return 0;*/


    
//    return 0;

//}


