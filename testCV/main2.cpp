/*参考：http://blog.csdn.net/jinshengtao/article/details/17954427 */

#include <cv.h>  
#include <highgui.h>  
#include <cvaux.h>  
#include <ml.h>    
#include <iostream>  
#include <vector>  
#include "Plate.h"  

#define HORIZONTAL    1  
#define VERTICAL    0  
using namespace std;  
using namespace cv;

CvANN_MLP  ann;  
const char strCharacters[] = {'0','1','2','3','4','5','6','7','8','9','B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'};  
const int numCharacters=30; 

  
//对minAreaRect获得的最小外接矩形，用纵横比进行判断  
bool verifySizes(RotatedRect mr)  
{  
    float error=0.4f;  
    //Spain car plate size: 52x11 aspect 4,7272  
    float aspect=4.7272f;  
    //Set a min and max area. All other patchs are discarded  
    int min= (int)(15*aspect*15); // minimum area  
    int max= (int)(125*aspect*125); // maximum area  
    //Get only patchs that match to a respect ratio.  
    float rmin= aspect-aspect*error;  
    float rmax= aspect+aspect*error;  
  
    int area= (int)(mr.size.height * mr.size.width);  
    float r= (float)(mr.size.width / (float)mr.size.height);  
    if(r<1)  
        r= (float)(mr.size.height / (float)mr.size.width);  
  
    if(( area < min || area > max ) || ( r < rmin || r > rmax )){  
        return false;  
    }else{  
        return true;  
    }  
  
}  
  
//直方图均衡化  
Mat histeq(Mat in)  
{  
    Mat out(in.size(), in.type());  
    if(in.channels()==3){  
        Mat hsv;  
        vector<Mat> hsvSplit;  
        cvtColor(in, hsv, CV_BGR2HSV);  
        split(hsv, hsvSplit);  
        equalizeHist(hsvSplit[2], hsvSplit[2]);  
        merge(hsvSplit, hsv);  
        cvtColor(hsv, out, CV_HSV2BGR);  
    }else if(in.channels()==1){  
        equalizeHist(in, out);  
    }  
  
    return out;  
  
}  

/**
  * @brief  抠图，从图片里抠出车牌
  *
  * @param  Img: 抠图片
  *
  * @retval 
  *
  */
Mat matting(char *fileRead, char *fileWrite) 
{  
	namedWindow("LPR", CV_WINDOW_NORMAL);
    

    Mat img_gray = imread(fileRead,CV_LOAD_IMAGE_GRAYSCALE);  
    Mat input = imread(fileRead);  
    //char res[20];  
  
    //apply a Gaussian blur of 5 x 5 and remove noise  
    blur(img_gray,img_gray,Size(5,5));
  
    //Finde vertical edges. Car plates have high density of vertical lines  
    Mat img_sobel;  
    Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);//xorder=1,yorder=0,kernelsize=3 
  
    //apply a threshold filter to obtain a binary image through Otsu's method  
    Mat img_threshold;  
    threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);

  
    //Morphplogic operation close:remove blank spaces and connect all regions that have a high number of edges  
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3) );  
    morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);  
	
  
     //Find 轮廓 of possibles plates  
    vector< vector< Point> > contours;  
    findContours(img_threshold,  
        contours, // a vector of contours  
        CV_RETR_EXTERNAL, // 提取外部轮廓  
        CV_CHAIN_APPROX_NONE); // all pixels of each contours  
  
    //Start to iterate to each contour founded  
    vector<vector<Point> >::iterator itc= contours.begin();  
    vector<RotatedRect> rects;  
  
    //Remove patch that are no inside limits of aspect ratio and area.      
    while (itc!=contours.end()) {  
        //Create bounding rect of object  
        RotatedRect mr= minAreaRect(Mat(*itc));  
        if( !verifySizes(mr)){  
            itc= contours.erase(itc);  
        }else{  
            ++itc;  
            rects.push_back(mr);  
        }  
    }  
  
    // Draw blue contours on a white image  
    cv::Mat result;  
    //input.copyTo(result);  
    //cv::drawContours(result,contours,  
    //  -1, // draw all contours  
    //  cv::Scalar(0,0,255), // in blue  
    //  3); // with a thickness of 1  
  
  
	Mat grayResult;  

    for(unsigned int i=0; i < rects.size(); i++)  
    {  
        //For better rect cropping for each posible box  
        //Make floodfill algorithm because the plate has white background  
        //And then we can retrieve more clearly the contour box  
        circle(result, rects[i].center, 3, Scalar(0,255,0), -1);
		
        //获取宽度和高度中最小值  
        float minSize=(rects[i].size.width < rects[i].size.height)?rects[i].size.width:rects[i].size.height;  
        minSize=minSize-minSize*0.5f;  
        //初始化填充算法中心的五个点和边缘
        srand ( (unsigned)time(NULL) ); //初始化随机数发生器
        //初始化填充工具的参数和变量
        Mat mask;  
        mask.create(input.rows + 2, input.cols + 2, CV_8UC1);  
        mask= Scalar::all(0);  
        int loDiff = 30;  
        int upDiff = 30;  
        int connectivity = 4;  
        int newMaskVal = 255;  
        int NumSeeds = 10;  
        Rect ccomp;  
        int flags = connectivity + (newMaskVal << 8 ) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;  
        for(int j=0; j<NumSeeds; j++){  
            Point seed;  
            seed.x=(int)(rects[i].center.x+rand()%(int)minSize-(minSize/2));  
            seed.y=(int)(rects[i].center.y+rand()%(int)minSize-(minSize/2));  
            circle(result, seed, 1, Scalar(0,255,255), -1);  
            int area = floodFill(input, mask, seed, Scalar(255,0,0), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags);  
        }  
		//imshow("LPR", mask);
		//waitKey();

        //Check new floodfill mask match for a correct patch.  
        //Get all points detected for get Minimal rotated Rect  
        vector<Point> pointsInterest;  
        Mat_<uchar>::iterator itMask= mask.begin<uchar>();  
        Mat_<uchar>::iterator end= mask.end<uchar>();  
        for( ; itMask!=end; ++itMask)  
            if(*itMask==255)  
                pointsInterest.push_back(itMask.pos());  
  
        RotatedRect minRect = minAreaRect(pointsInterest);  
  
        if(verifySizes(minRect)){  
            // rotated rectangle drawing   
            Point2f rect_points[4]; minRect.points( rect_points );  
            for( int j = 0; j < 4; j++ )  
                line( result, rect_points[j], rect_points[(j+1)%4], Scalar(0,0,255), 1, 8 );      
  
            //Get rotation matrix  
            float r= (float)minRect.size.width / (float)minRect.size.height;  
            float angle=minRect.angle;      
            if(r<1)  
                angle=90+angle;  
            Mat rotmat= getRotationMatrix2D(minRect.center, angle,1);  
  
            //Create and rotate image  
            Mat img_rotated;  
            warpAffine(input, img_rotated, rotmat, input.size(), CV_INTER_CUBIC);  

            //Crop image  
            Size rect_size=minRect.size;  
            if(r < 1)  
                swap(rect_size.width, rect_size.height);  
            Mat img_crop;  
            getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);  

            Mat resultResized;  
            resultResized.create(33,144, CV_8UC3);  
            resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);  
			
            //Equalize croped image  
            //Mat grayResult;  
            cvtColor(resultResized, grayResult, CV_BGR2GRAY);   
            blur(grayResult, grayResult, Size(3,3));  
            grayResult=histeq(grayResult);
			imshow("LPR",grayResult);waitKey();
        /*  if(1){  
                stringstream ss(stringstream::in | stringstream::out); 
                ss << "haha" << "_" << i << ".jpg"; 
                imwrite(ss.str(), grayResult); 
            }*/  
            //output.push_back(Plate(grayResult,minRect.boundingRect()));  
        }  
    }  


	imwrite(fileWrite, grayResult);
    waitKey(0);  


    return grayResult;  
}




//这里应该是来验证 框住的是不是一个数字。可能是根据长宽比例。
bool verifySizes(Mat r){  
    //Char sizes 45x77  
    float aspect=45.0f/77.0f;  
    float charAspect= (float)r.cols/(float)r.rows;  
    float error=0.35f;  
    float minHeight=15;  
    float maxHeight=28;  
    //We have a different aspect ratio for number 1, and it can be ~0.2  
    float minAspect=0.2f;  
    float maxAspect=aspect+aspect*error;  
    //area of pixels  
    float area=(float)countNonZero(r);  
    //bb area  
    float bbArea=(float)(r.cols*r.rows);  
    //% of pixel in area  
    float percPixels=area/bbArea;  
  
    /*if(DEBUG) 
    cout << "Aspect: "<< aspect << " ["<< minAspect << "," << maxAspect << "] "  << "Area "<< percPixels <<" Char aspect " << charAspect  << " Height char "<< r.rows << "\n";*/  
    if(percPixels < 0.8 && charAspect > minAspect && charAspect < maxAspect && r.rows >= minHeight && r.rows < maxHeight)  
        return true;  
    else  
        return false;  
  
}  
 
//这里主要是进行旋转变换 和 尺寸放缩。
Mat preprocessChar(Mat in){  
    //Remap image  
    int h=in.rows;  
    int w=in.cols;  
    int charSize=20;    //统一每个字符的大小  

    Mat transformMat=Mat::eye(2,3,CV_32F); 

    int m=max(w,h);  
    transformMat.at<float>(0,2)=(float)(m/2 - w/2);  
    transformMat.at<float>(1,2)=(float)(m/2 - h/2);  
  
    Mat warpImage(m,m, in.type());   //这里主要是进行旋转变换 和 尺寸放缩。
    warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0) );  
  
    Mat out;  
    resize(warpImage, out, Size(charSize, charSize) );   
  
    return out;  
}  
 
//create the accumulation histograms,img is a binary image, t is 水平或垂直  
Mat ProjectedHistogram(Mat img, int t)  
{  
    int sz=(t)?img.rows:img.cols;  
    Mat mhist=Mat::zeros(1,sz,CV_32F);  
  
    for(int j=0; j<sz; j++){  
        Mat data=(t)?img.row(j):img.col(j);  
        mhist.at<float>(j)=(float)countNonZero(data);    //统计这一行或一列中，非零元素的个数，并保存到mhist中  
    }  
  
    //Normalize histogram  
    double min, max;  
    minMaxLoc(mhist, &min, &max);  
  
    if(max>0)  
        mhist.convertTo(mhist,-1 , 1.0f/max, 0);//用mhist直方图中的最大值，归一化直方图  
  
    return mhist;  
}  
 

Mat features(Mat in, int sizeData){  
    //Histogram features  
    Mat vhist=ProjectedHistogram(in,VERTICAL);  
    Mat hhist=ProjectedHistogram(in,HORIZONTAL);  
  
    //Low data feature  
    Mat lowData;  
    resize(in, lowData, Size(sizeData, sizeData) );  
  
    //Last 10 is the number of moments components  
    int numCols=vhist.cols+hhist.cols+lowData.cols*lowData.cols;  
  
    Mat out=Mat::zeros(1,numCols,CV_32F);  
    //Asign values to feature,ANN的样本特征为水平、垂直直方图和低分辨率图像所组成的矢量  
    int j=0;  
    for(int i=0; i<vhist.cols; i++)  
    {  
        out.at<float>(j)=vhist.at<float>(i);  
        j++;  
    }  
    for(int i=0; i<hhist.cols; i++)  
    {  
        out.at<float>(j)=hhist.at<float>(i);  
        j++;  
    }  
    for(int x=0; x<lowData.cols; x++)  
    {  
        for(int y=0; y<lowData.rows; y++){  
            out.at<float>(j)=(float)lowData.at<unsigned char>(x,y);  
            j++;  
        }  
    }  
      
    return out;  
}  
  
//这个是进行预测的函数。传入的是定义好的特征。预测也是封装在函数中进行了。传入的是不是和训练的是一样的？？？？
int classify(Mat f){  
    
	
	int result=-1;  
    Mat output(1, 30, CV_32FC1); //西班牙车牌只有30种字符 。1行30列的Mat 矩阵。CV_32FC1 表示每个元素都是float？？

    ann.predict(f, output);		 //这里进行预测。输出在 1行30列的Mat 矩阵中。ANN 自带的，不用关心。

    Point maxLoc;  
    double maxVal;  
    minMaxLoc(output, 0, &maxVal, 0, &maxLoc);  //寻找最大值和最小值的位置。0 的位置表示该参数不需要。

    //We need know where in output is the max val, the x (cols) is the class.   
    return maxLoc.x;							//x是它在第几列的坐标。这里要注意 小心出错。
}  
    

//训练的过程。重要。将ANN的训练函数封装在这个里面了。
//TrainData 是全部训练样本组成的矩阵，每一行是一个样本，列数反映每一个样本提取到的特征数目。
//classes 是全部的样本标签 和TrainData 行数相同，只有一个列。
//nlayers 是隐含层神经元的数目。
void train(Mat TrainData, Mat classes, int nlayers){  

    Mat layers(1,3,CV_32SC1);  

    layers.at<int>(0)= TrainData.cols;  // 输入层为675行，但是这里怎么是用的列？？？？？？？？？？
    layers.at<int>(1)= nlayers;  
    layers.at<int>(2)= 30;			   //因为有30个字符

	//int i = TrainData.cols;
	//int j = nlayers;
	//Mat layerSizes=(Mat_<int>(1,5) << 5,2,2,2,5); 
	//cout << layerSizes << endl;
	//ann.create(layerSizes,CvANN_MLP::SIGMOID_SYM);

    ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);  
  
    //Prepare trainClases  
    //Create a mat with n trained data by m classes

	//创建的这个矩阵包含了675行和30列。675行反映总共675个样例。30列表示的是有30种分类。
	//
    Mat trainClasses;  
    trainClasses.create( TrainData.rows, 30, CV_32FC1 );     
	
	//将创建出来的这个矩阵用0 和 1 填充了。每一行的1所在的列表示 该行样本的标签。
    for( int i = 0; i <  trainClasses.rows; i++ )  
    {  
        for( int k = 0; k < trainClasses.cols; k++ )  
        {  
            //If class of data i is same than a k class  
            if( k == classes.at<int>(i) )  
                trainClasses.at<float>(i,k) = 1;  
            else  
                trainClasses.at<float>(i,k) = 0;  
        }  
    } 

	//1 行n列的矩阵Mat。n 等于训练集矩阵的列数。。全部用1填充。
    Mat weights( 1, TrainData.rows, CV_32FC1, Scalar::all(1) );  
  
    //Learn classifier  
	//参数。TrainData：675*n；trainClasses：675*30；weight：1*n。
    ann.train( TrainData, trainClasses, weights );  
}  
  

int main()
{  
	
	char *fileRead = "chepai.png";
	char *fileWrite = "chepai_2.png";

	//这个处理应该是先将车牌抠出来，用于下面的神经网络分类。
	matting(fileRead,fileWrite);


	//下面是神经网络部分
    Mat input = imread(fileWrite,CV_LOAD_IMAGE_GRAYSCALE);  
    

	Plate mplate;  

    //Read file storage.  
    FileStorage fs;  
    fs.open("OCR.xml", FileStorage::READ);		//这里的文件是提前计算好的，将样本的特征提取出来组成的一个矩阵。

    Mat TrainingData;  
    Mat Classes;  

	//这样可以将数据标签相对应的数据写进这个矩阵Mat
    fs["TrainingDataF15"] >> TrainingData;		//将训练集组成的特征提取出来
    fs["classes"] >> Classes;					//训练集特征对应的标签。是个一维列矩阵。
	

    //训练神经网络。。。有一个出来的模型  
    train(TrainingData, Classes, 10);				//这里是训练的过程，很重要。将训练封装到一个函数中进行。
	//ann.save("20160625.xml");	//将训练结果保存，下次可以直接用
	ann.load("20160625.xml");    //也可以直接加载之前训练好的模型。


	//下面的是预测的过程
    //dealing image and save each character image into vector<CharSegment>  
    //Threshold input image      
	Mat img_threshold;  
    threshold(input, img_threshold, 60, 255, CV_THRESH_BINARY_INV);  
  
    Mat img_contours;  
    img_threshold.copyTo(img_contours);  


    //Find contours of possibles characters  这里是在寻找轮廓。

    vector< vector< Point> > contours;  
    findContours(img_contours,  
        contours, // a vector of contours  
        CV_RETR_EXTERNAL, // retrieve the external contours  
        CV_CHAIN_APPROX_NONE); // all pixels of each contours  

	//Start to iterate to each contour founded  
    vector<vector<Point> >::iterator itc= contours.begin();    
    //Remove patch that are no inside limits of aspect ratio and area.      
    while (itc!=contours.end()) {  
  
        //Create bounding rect of object  
        Rect mr= boundingRect(Mat(*itc));     //寻找每一个被框住的矩形。
        //rectangle(result, mr, Scalar(255,0,0),2);  
        //Crop image  
        Mat auxRoi(img_threshold, mr);		//这个很有用，只是将框住的部分 复制到另一个Mat 中去了。
        
		if(verifySizes(auxRoi)){			//if 条件中将不满足长宽比例的都给过滤掉了。


            auxRoi=preprocessChar(auxRoi);   //这一步是做什么预处理？  //这里主要是进行旋转变换 和 尺寸放缩。
  
            //对每一个小方块，提取定义的全部特征  
            Mat f=features(auxRoi,15);			//这个Mat f 是一个1维的行向量，包含该图片要提取的全部特征。！！！！

            //For each segment feature Classify  预测也是封装在函数中进行了
            int character=classify(f);         //这里是进行预测。将每一个方块的特征传进去。 int 返回的其实是个分类的索引。

            mplate.chars.push_back(strCharacters[character]);  
            mplate.charsPos.push_back(mr);  //若这一句一直不执行，则在对象内部成员属性为0。会出现向量vector 越界错误。
            
        }  
        ++itc;  
    } 


    string licensePlate=mplate.str();  
    cout<<licensePlate<<endl;  
      
	
	system("pause");
    return 0;  
}  