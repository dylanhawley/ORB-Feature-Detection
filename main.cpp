#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    // Read train image
    std::string train_img_filename(argv[1]);
    std::cout << "Reading reference image : " << train_img_filename << std::endl;
    cv::Mat train_img = cv::imread(train_img_filename);

    // Read image to be analyzed
    std::string query_img_filename(argv[2]);
    std::cout << "Reading query image : " << query_img_filename << std::endl;
    cv::Mat query_img = cv::imread(query_img_filename);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    std::vector<cv::KeyPoint> train_img_keypoints, query_img_keypoints;
    cv::Mat train_img_descriptors, query_img_descriptors;

    orb->detectAndCompute(train_img, cv::noArray(), train_img_keypoints, train_img_descriptors);
    orb->detectAndCompute(query_img, cv::noArray(), query_img_keypoints, query_img_descriptors);

    cv::BFMatcher bf(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    bf.match(train_img_descriptors, query_img_descriptors, matches);

    // Sort for good matches
    std::sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> good_matches;
    for(int i = 0; i < 10; i++)
        good_matches.push_back(matches[i]);

    cv::Mat img_matches;
    cv::drawMatches(train_img, train_img_keypoints, query_img, query_img_keypoints,
                    good_matches, img_matches);
    cv::namedWindow("Matches", cv::WINDOW_AUTOSIZE);
    cv::imshow("Matches", img_matches);
    cv::waitKey();

    return 0;
}