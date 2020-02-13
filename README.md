# ORB_Feature

1、说明
此程序是将ORB-SLAM2(https://github.com/raulmur/ORB_SLAM2)代码中的ORB特征提取部分拿出来单独测试，
并结合GMS(https://github.com/JiawangBian/GMS-Feature-Matcher)算法进行特征匹配。

2、系统和依赖库：
使用ubunttu16.04和OpenCV3.3.1进行的代码测试

3、编译
./build.sh

4、运行
特征提取：./bin/orb_modify TUM2.yaml data/1.png 
特征提取与匹配：./bin/orb_matcher TUM2.yaml data/1.png data/2.png
