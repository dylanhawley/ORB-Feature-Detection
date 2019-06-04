#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    // Read reference image
    std::string refFilename("reference_group.png");
    std::cout << "Reading reference image : " << refFilename << std::endl;
    cv::Mat imReference = cv::imread(refFilename);

    // Read image to be analyzed
    std::string imFilename("modified_group.png");
    std::cout << "Reading image to align : " << imFilename << std::endl;
    cv::Mat im = cv::imread(imFilename);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> keypoints1;
    cv::Mat descriptors1;
    orb->detectAndCompute(imReference, cv::noArray(), keypoints1, descriptors1);

    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors2;
    orb->detectAndCompute(im, cv::noArray(), keypoints2, descriptors2);

    cv::drawKeypoints(im,keypoints2,imReference);
    cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original", imReference);
    cv::waitKey();

    return 0;
}