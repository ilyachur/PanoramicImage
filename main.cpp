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
    int startFrom = 0;

    for(int i = 0; i < result.cols; i++) {
        //if (result.at<Vec3b>((int)((result.rows - 1) / 2), i) != Vec3b(0, 0, 0)) {
        if (result.at<Vec3b>(0, i) != Vec3b(0, 0, 0) && result.at<Vec3b>(result.rows - 1, i) != Vec3b(0, 0, 0)) {
            startFrom = i;
            break;
        }
    }

    for(int i = result.cols - 1; i >= 0; i--) {
        //if (result.at<Vec3b>((int)((result.rows - 1) / 2), i) != Vec3b(0, 0, 0)) {
        if (result.at<Vec3b>(0, i) != Vec3b(0, 0, 0) && result.at<Vec3b>(result.rows - 1, i) != Vec3b(0, 0, 0)) {
            realSize = i;
            break;
        }
    }

    cv::Rect cutImage(startFrom,0, realSize - startFrom, result.rows);
    return result(cutImage);
}

Mat imagesGlue(Mat& image1, Mat& image2, int defaultCompareSize, bool left = false) {
    cv::Mat result;
    std::cout << defaultCompareSize << std::endl;

    Mat gray_image1;
    Mat gray_image2;
    std::cout << "left " << left << std::endl;

    int startX = image2.cols - defaultCompareSize;
    cv::Rect prevImageRect(0,0, min(image1.cols, defaultCompareSize), image1.rows);
    std::cout << " Usage " << image1.cols << " "<< prevImageRect << std::endl;
    Mat previouseImage(image1, prevImageRect);

    cv::Rect nextImageRect(max(0, startX),0, image2.cols - max(0, startX), image2.rows);
    std::cout << " Usage " << image2.cols << " "<<  nextImageRect << std::endl;
    Mat nextImage(image2, nextImageRect);

    // Convert to Grayscale
    cvtColor( previouseImage, gray_image1, CV_RGB2GRAY );
    cvtColor( nextImage, gray_image2, CV_RGB2GRAY );

    //imshow("first image", image2);
    //imshow("second image",image1);

    if( !gray_image1.data || !gray_image2.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return Mat(); }

    cv::Mat scaleResult1;
    cv::resize(gray_image1, scaleResult1, cv::Size(gray_image1.cols / 10, gray_image1.rows / 10));
    cv::Mat scaleResult2;
    cv::resize(gray_image2, scaleResult2, cv::Size(gray_image2.cols / 10, gray_image2.rows / 10));
    imshow( "first image", scaleResult1 );
    imshow( "second image", scaleResult2 );
    //waitKey(0);

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector( minHessian );

    std::vector< KeyPoint > keypoints_object, keypoints_scene;

    detector.detect( gray_image1, keypoints_object );
    detector.detect( gray_image2, keypoints_scene );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_object, descriptors_scene;

    extractor.compute( gray_image1, keypoints_object, descriptors_object );
    extractor.compute( gray_image2, keypoints_scene,  descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    BFMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_object.rows; i++ )
    {
        if( matches[i].distance < 3*min_dist )
            good_matches.push_back( matches[i]);
    }
    std::vector< Point2f > obj;
    std::vector< Point2f > scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
        if (left)
            obj.back().x += image1.cols;
        else
            scene.back().x += image2.cols - defaultCompareSize;
    }

    // Find the Homography Matrix
    Mat H = findHomography( obj, scene, CV_RANSAC );
    if (left)
        H = findHomography( scene, obj, CV_RANSAC );

    // Use the Homography Matrix to warp the images
    if (left) {
        warpPerspective(image2, result, H, Size((image1.cols+image2.cols)*5, image2.rows));

        cv::Mat scaleResult3;
        cv::resize(result, scaleResult3, cv::Size(result.cols / 10, result.rows / 10));
        imshow( "res31", scaleResult3 );
        image1.copyTo(result(Rect(image1.cols, 0, image1.cols, image1.rows)));
        cv::resize(result, scaleResult3, cv::Size(result.cols / 10, result.rows / 10));
        imshow( "res32", scaleResult3 );
    } else {
        warpPerspective(image1, result, H, Size((image1.cols+image2.cols)*5, image1.rows));
        Mat half (result, Rect(0,0,image2.cols,image2.rows));
        cv::Mat scaleResult3;
        cv::resize(result, scaleResult3, cv::Size(result.cols / 10, result.rows / 10));
        imshow( "res31", scaleResult3 );
        image2.copyTo(half);
        cv::resize(result, scaleResult3, cv::Size(result.cols / 10, result.rows / 10));
        imshow( "res32", scaleResult3 );
    }

    waitKey(0);
    return result;
}

int main( int argc, char** argv ) {
    if( argc < 3 ) {
        std::cout << " Usage: PanoramicImage <img1> ... <imgN>" << std::endl;
        return -1;
    }
    cv::Mat result;
    if ((argc - 1) % 2 != 0) {
        result = imread(argv[int((argc - 1) / 2) + 1]);
    }
    int defaultCompareSize = 0;
    int count = 0;
    for (int i = (argc - 1) / 2; i >= 1; i--) {
        std::cout << "images" << std::endl;
        std::cout << i << std::endl;
        std::cout << argc - i << std::endl;
        std::cout << "images" << std::endl;
        Mat image1 = imread(argv[i]);
        if (defaultCompareSize == 0)
            defaultCompareSize = image1.cols;
        if (result.data) {
            count++;
            std::cout << "count " << count << std::endl;
            if (count < 2) {
            Mat tmpRes = imagesGlue(result, image1, defaultCompareSize, true);
            findRealSize(tmpRes).copyTo(result);
            } else {
                Mat tmpRes = imagesGlue(image1, result, defaultCompareSize);
                findRealSize(tmpRes).copyTo(result);
            }

            if (!result.data)
                return -1;
            image1 = result;
        }
        Mat image2 = imread(argv[argc - i]);
        if (!result.data) {
            Mat tmpRes = imagesGlue(image2, image1, defaultCompareSize, true);
            findRealSize(tmpRes).copyTo(result);
        } else {
            Mat tmpRes = imagesGlue(image2, image1, defaultCompareSize);
            findRealSize(tmpRes).copyTo(result);
        }
        if (!result.data)
            return -1;
    }

    /*cv::Mat scaleResult;
    cv::resize(result, scaleResult, cv::Size(result.cols / 15, result.rows / 15));
    imshow( "Result", scaleResult );
    waitKey(0);*/
    imwrite( "out.png", result );

    return 0;
}
