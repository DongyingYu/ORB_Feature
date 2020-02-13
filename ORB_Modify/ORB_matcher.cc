#include <iostream>
#include <string>

#include "gms_matcher.h"
#include "ORB_modify.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main ( int argc, char** argv )
{
    if ( argc != 4 )
    {
        cout<<"usage: ./orb_matcher path_to_settings path_to_image1 path_to_image2"<<endl;
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

    Mat img1 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    Mat img2 = imread ( argv[3], CV_LOAD_IMAGE_COLOR );

    ORB_modify ORB_left(strSettingsFile);
    ORB_modify ORB_right(strSettingsFile);
    ORB_left.ORB_feature(img1);
    ORB_right.ORB_feature(img2);

    vector<DMatch> matches_all, matches_gms;
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(ORB_left.mDescriptors, ORB_right.mDescriptors, matches_all);

	// GMS filter
	std::vector<bool> vbInliers;
	gms_matcher gms(ORB_left.mvKeysUn, img1.size(), ORB_right.mvKeysUn, img2.size(), matches_all);

	int num_inliers = gms.GetInlierMask(vbInliers, false, false);
	cout << "Get total " << num_inliers << " matches." << endl;

	// collect matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}

	// draw matching
	Mat show =gms.DrawInlier(img1, img2, ORB_left.mvKeysUn, ORB_right.mvKeysUn, matches_gms, 2);

    imwrite("./result/ORB_matcher.png", show);
	imshow("ORB_matcher", show);

    cv::waitKey(0);
    return 0;
}