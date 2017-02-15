#ifndef locator_h  
#define locator_h  
  
#include <string.h>  
#include <vector>  
  
#include <cv.h>  
#include <highgui.h>  
#include <cvaux.h>  
  
using namespace std;  
using namespace cv;  
  
class Locator{  
    public:  
        Locator();  
        Locator(char* imgpath);
		void shapeDetect();
		vector<Mat> colorDetect();
        string str(); 
	private:
		char* path;
		RNG rng; 
              
};  
  
#endif  