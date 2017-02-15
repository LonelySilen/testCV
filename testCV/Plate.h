#ifndef Plate_h  
#define Plate_h  
  
#include <string.h>  
#include <vector>  
  
#include <cv.h>  
#include <highgui.h>  
#include <cvaux.h>  
  
using namespace std;  
using namespace cv;  
  
class Plate{  
    public:  
        Plate();  
        Plate(Mat img, Rect pos);  
        string str();  
        Rect position;  
        Mat plateImg;  
        vector<char> chars;  
        vector<Rect> charsPos;          
};  
  
#endif  