#include <stdio.h>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

Mat findRealSize(Mat& result) {
    //std::cout << "result = \n" << result << std::endl << std::endl;
    int realSize = result.cols;
    for(int i = 0; i < result.cols; i++) {
        if (result.at<Vec3b>(0, i) == Vec3b(0, 0, 0)) {
            int j = i;
            for(j = i; j < result.cols; j++) {
                if (result.at<Vec3b>(0, j) != Vec3b(0, 0, 0))
                    break;
            }
            if (j == result.cols) {
                realSize = i;
                break;
            }
        }
        if (result.at<Vec3b>(result.rows - 1, i) == Vec3b(0, 0, 0)) {
            int j = i;
            for(j = i; j < result.cols; j++) {
                if (result.at<Vec3b>(result.rows - 1, j) != Vec3b(0, 0, 0))
                    break;
            }
            if (j == result.cols) {
                realSize = i;
                break;
            }
        }
    }
    //warpPerspective(result,result,H,cv::Size(realSize,result.rows));
    cv::Rect cutImage(0,0, realSize, result.rows);
    return result(cutImage);
}

Mat imagesGlue(Mat& image1, Mat& image2, int defaultCompareSize) {
    cv::Mat result;

    int startX = image1.cols - defaultCompareSize*2;
    if (startX < 0)
        startX = 0;
    cv::Rect prevImageRect(max(0, startX),0, image1.cols - max(0, startX), image1.rows);
    std::cout << " Usage " << prevImageRect << std::endl;
    Mat previouseImage(image1, prevImageRect);

    cv::Rect nextImageRect(0,0, min(image2.cols, defaultCompareSize * 2), image2.rows);
    std::cout << " Usage " << nextImageRect << std::endl;
    Mat nextImage(image2, nextImageRect);

    Mat gray_image1;
    Mat gray_image2;

    // Convert to Grayscale
    cvtColor( previouseImage, gray_image1, CV_RGB2GRAY );
    cvtColor( nextImage, gray_image2, CV_RGB2GRAY );

    if( !gray_image1.data || !gray_image2.data ) {
        std::cout<< " --(!) Error reading images " << std::endl;
        return Mat();
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector(minHessian);
    std::vector< KeyPoint > keypoints_object, keypoints_scene;

    detector.detect(gray_image2, keypoints_object);
    detector.detect(gray_image1, keypoints_scene);

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    extractor.compute( gray_image2, keypoints_object, descriptors_object );
    extractor.compute( gray_image1, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ ) {
        double dist = matches[i].distance;
        if( dist < min_dist )
            min_dist = dist;
        if( dist > max_dist )
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ ) {
        if( matches[i].distance < 3*min_dist ) {
            good_matches.push_back( matches[i]);
        }
    }

    std::vector< Point2f > obj;
    std::vector< Point2f > scene;

    for( int i = 0; i < good_matches.size(); i++ ) {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    // Find the Homography Matrix
    Mat H = findHomography( obj, scene, CV_RANSAC );
    // Use the Homography Matrix to warp the images
    warpPerspective(image2,result,H,cv::Size((image1.cols+image2.cols) * 5,image2.rows));
    cv::Mat half(result,cv::Rect(0,0,image1.cols,image1.rows));
    image1.copyTo(half);

    return result;
}

int main( int argc, char** argv ) {
    if( argc < 3 ) {
        std::cout << " Usage: PanoramicImage <img1> ... <imgN>" << std::endl;
        return -1;
    }
    cv::Mat result;


    cv::Mat *results = new Mat[(argc + 1) / 3];
    std::cout << (argc + 1) / 3 << " " << results << std::endl;

    int defaultCompareSize = 0;
    for (int iter = 0; iter < (argc + 1) / 3; iter++) {
        for (int i = iter*3 + 2; i < (iter+1)*3 + 1 && i < argc; i++) {
            // Load the images
            Mat image1 = results[iter];
            if(!image1.data)
                image1 = imread( argv[i - 1] );
            if (defaultCompareSize == 0)
                defaultCompareSize = image1.cols;
            Mat image2= imread( argv[i] );
            Mat iterRes = imagesGlue(image1, image2, defaultCompareSize);
            findRealSize(iterRes).copyTo(results[iter]);
            if (!results[iter].data)
                return -1;
        }
    }
    imagesGlue(results[0], results[1], defaultCompareSize).copyTo(result);
    //results[1].copyTo(result);
    cv::Mat scaleResult;
    cv::resize(result, scaleResult, cv::Size(result.cols / 6, result.rows / 6));
    imshow( "Result", scaleResult );
    waitKey(0);

    return 0;
}
