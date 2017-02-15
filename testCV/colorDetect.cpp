#include "colorDetect.h"

colorDetect::colorDetect(void)
{
	minDist = 50;
	target[2]= 150;
	target[1]= 150;
	target[0]= 150;
}
void colorDetect::SetMinDistance(int dist)
{
	minDist = dist;
}

void colorDetect::SetTargetColor(uchar red,uchar green,uchar blue)
{
	target[2]= red;
	target[1]= green;
	target[0]= blue;
}

void colorDetect::SetTargetColor(Vec3b color)
{
	target = color;
}

Mat colorDetect::process(const Mat& image)
{
    Mat ImageLab=image.clone();
    result.create(image.rows,image.cols,CV_8U);

    //��imageת��ΪLab��ʽ�洢��ImageLab��
    cvtColor(image,ImageLab,CV_BGR2Lab);
    //��Ŀ����ɫ��BGRת��ΪLab
    Mat temp(1,1,CV_8UC3);
    temp.at<Vec3b>(0,0)=target;//������һ��1*1����ʱͼ����Ŀ����ɫ���
    cvtColor(temp,temp,CV_BGR2Lab);
    target=temp.at<Vec3b>(0,0);//�ٴ���ʱͼ���Lab��ʽ��ȡ��Ŀ����ɫ

    // ���������õĵ�����
    Mat_<Vec3b>::iterator it=ImageLab.begin<Vec3b>();
    Mat_<Vec3b>::iterator itend=ImageLab.end<Vec3b>();
    Mat_<uchar>::iterator itout=result.begin<uchar>();
    while(it!=itend)
    {
        //������ɫֵ֮�����ļ���
        int dist=static_cast<int>(norm<int,3>(Vec3i((*it)[0]-target[0],
            (*it)[1]-target[1],(*it)[2]-target[2])));
        if(dist<minDist)
            (*itout)=255;
        else
            (*itout)=0;
        it++;
        itout++;
    }
    return result;
}