#include <string>

#include "ORB_modify.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main ( int argc, char** argv )
{
    if ( argc != 3 )
    {
        cout<<"usage: ./orb_modify path_to_settings path_to_image "<<endl;
        return 1;
    }

    //Check settings file
    const string strSettingsFile = argv[1];
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }

    ORB_modify ORB(strSettingsFile);
    Mat img = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    ORB.ORB_feature(img);
    cv::waitKey(0);

    return 0;
}