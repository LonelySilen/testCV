#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
class colorDetect{
private:
    int minDist; //minium acceptable distance
    Vec3b target;//target color;    
    Mat result; //the result
public:
    colorDetect(void);
    void SetMinDistance(int dist);
    void SetTargetColor(uchar red,uchar green,uchar blue);
    void SetTargetColor(Vec3b color); //set the target color
    Mat process(const Mat& image); //main process
};


