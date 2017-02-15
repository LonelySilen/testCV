/***************************************************************************** 
*   Number Plate Recognition using SVM and Neural Networks 
*****************************************************************************/  
  
#include "locator.h"  
  
Locator::Locator(){  
	path = "";
}  
  
Locator::Locator(char* imgpath){  
	path = imgpath;
   

}  
  
string Locator::str(){  
    string result="";  
   
    return result;  
} 

void Locator::shapeDetect(){
	Mat srcImage=imread(path);  
    //imshow("a",srcImage);  
    int i,j;
    int cPointR,cPointG,cPointB,cPoint;//currentPoint;  
    Mat resizeImage;  
    resize(srcImage,resizeImage,Size(400,300));  
    Mat grayImage;  
    cvtColor(resizeImage,grayImage, CV_BGR2GRAY);  
    Mat medianImage;  
    medianBlur(grayImage,medianImage,3); //最后一个参数需要为奇数  
    Mat sobelImage;  
    //参数为：源图像，结果图像，图像深度，x方向阶数，y方向阶数，核的大小，尺度因子，增加的值    
  
    Sobel(medianImage,sobelImage,CV_8U,1,0,3,0.4,128);    
    Mat normalizeImage;  
    normalize(sobelImage,normalizeImage,255,0,CV_MINMAX);  
    Mat binaryImage;  
    threshold(normalizeImage,binaryImage, 100, 255, THRESH_BINARY_INV );    
    Mat closeImage;  
    //morphologyEx(binaryImage,closeImage,MORPH_CLOSE,Mat(3,1,CV_8U),Point(0,0),10);  //闭运算  
    Mat openImage(closeImage.rows,closeImage.cols,CV_8UC1);  
    //morphologyEx(closeImage,openImage,MORPH_OPEN,Mat(3,3,CV_8U),Point(0,0),1);   //开运算  
    //  erode(openImage,openImage,Mat(3,3,CV_8U),Point(-1,-1),10);  
    dilate(binaryImage,openImage,Mat(3,3,CV_8U),Point(-1,-1),6);  
    /* 
    Mat rgbImage; 
    cvtColor(openImage,rgbImage, CV_GRAY2BGR); 
    */  
    //cvtColor(openImage,openImage, CV_BGR2GRAY);  
    //vector<vector<Point> > contours;  
    //vector<Vec4i> hierarchy;  
    //openImage=imread("test.png");  
    imshow("openImage",openImage);  
    /// Detect edges using canny  
    //  Canny( src_gray, canny_output, thresh, thresh*2, 3 );  
    /// Find contours  
    /*  Mat thresholdImage; 
 
    cvtColor(openImage,openImage, CV_BGR2GRAY); 
    threshold( openImage,thresholdImage,127, 255, THRESH_BINARY ); 
    openImage=thresholdImage;*/  
  
  
    vector<vector<Point> > contours;  
    vector<Vec4i> hierarchy;  
    findContours(openImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );  
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );  
    for( int i = 0; i < contours.size(); i++ )  
    {    
        //使用边界框的方式    
        CvRect aRect =  boundingRect(contours[i]);  
        int tmparea=aRect.height*aRect.height;    
        if (((double)aRect.width/(double)aRect.height>2)&& ((double)aRect.width/(double)aRect.height<6)&& tmparea>=500&&tmparea<=25000)    
        {    
            rectangle(resizeImage,cvPoint(aRect.x,aRect.y),cvPoint(aRect.x+aRect.width ,aRect.y+aRect.height),color,2);    
            //cvDrawContours( dst, contours, color, color, -1, 1, 8 );   
			Mat rs;
			resizeImage(aRect).copyTo(rs);
			imshow("result"+i,rs); 
        }    
    }   
  
    imshow("contour",resizeImage); 
}

vector<Mat> Locator::colorDetect(){

	vector<Mat> rs_vec;

	Mat srcImage=imread(path);  
    Mat srcShowImage;  
    srcImage.copyTo(srcShowImage);  
    //imshow("a",srcImage);  
    int i,j;  
    int cPointB,cPointG,cPointR;  
    for(i=1;i<srcImage.rows;i++)  
        for(j=1;j<srcImage.cols;j++)  
        {  
            cPointB=srcImage.at<Vec3b>(i,j)[0];  
            cPointG=srcImage.at<Vec3b>(i,j)[1];  
            cPointR=srcImage.at<Vec3b>(i,j)[2];  
            if(cPointB>80&cPointR<80&cPointG<80)    //提取蓝色，将该区域设置为黑色  
            {  
                srcImage.at<Vec3b>(i,j)[0]=0;  
                srcImage.at<Vec3b>(i,j)[1]=0;  
                srcImage.at<Vec3b>(i,j)[2]=0;  
            }  
  
            else if(cPointB>200&cPointR>200&cPointG>200)  //提取白色，将其设置为黑色  
            {  
                srcImage.at<Vec3b>(i,j)[0]=0;  
                srcImage.at<Vec3b>(i,j)[1]=0;  
                srcImage.at<Vec3b>(i,j)[2]=0;  
            }  
  
            else  
            {  
                srcImage.at<Vec3b>(i,j)[0]=255;  
                srcImage.at<Vec3b>(i,j)[1]=255;  
                srcImage.at<Vec3b>(i,j)[2]=255;  
            }  
  
        }  
        cvtColor(srcImage,srcImage, CV_BGR2GRAY);    
        threshold(srcImage,srcImage,127, 255,CV_THRESH_BINARY);     
        //使用差分法，去掉不相关的区域。  
        for(i=1;i<srcImage.rows;i++)  
            for(j=1;j<srcImage.cols-1;j++)  
            {  
                srcImage.at<uchar>(i,j)=srcImage.at<uchar>(i,j+1)-srcImage.at<uchar>(i,j);  
  
            }  
  
            threshold(srcImage,srcImage,127, 255,CV_THRESH_BINARY_INV);//通过二值化的方式来取反。  
            //erode(srcImage,srcImage,Mat(5,5,CV_8U),Point(-1,-1),2);  //腐蚀  
            //  dilate(src,src,Mat(5,5,CV_8U),Point(-1,-1),2); //膨胀  
            //  morphologyEx(src,src,MORPH_OPEN,Mat(3,3,CV_8U),Point(-1,-1),1);   //开运算  
            //   morphologyEx(src,src,MORPH_CLOSE,Mat(3,3,CV_8U),Point(-1,-1),1);  //闭运算  
            erode(srcImage,srcImage,Mat(3,3,CV_8U),Point(-1,-1),5);  
            threshold(srcImage,srcImage,127,255,CV_THRESH_BINARY_INV);  
//            imshow("a",srcImage);  
            vector<vector<Point> > contours;  
            vector<Vec4i> hierarchy;  
            findContours(srcImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );  
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );  
            for( int i = 0; i < contours.size(); i++ )  
            {    
                //使用边界框的方式    
                CvRect aRect =  boundingRect(contours[i]);  
                int tmparea=aRect.height*aRect.height;    
                if (((double)aRect.width/(double)aRect.height>2)&& ((double)aRect.width/(double)aRect.height<6)&& tmparea>=2000&&tmparea<=25000)    
                {    
                    rectangle(srcShowImage,cvPoint(aRect.x,aRect.y),cvPoint(aRect.x+aRect.width ,aRect.y+aRect.height),color,2);    
                    //cvDrawContours( dst, contours, color, color, -1, 1, 8 );    
					Mat rs;
					srcShowImage(aRect).copyTo(rs);
//					imshow("result"+i,rs); 
					rs_vec.push_back(rs);
                }    
            }   
  
			
//			  imshow("da",srcShowImage); 
			return rs_vec;
}