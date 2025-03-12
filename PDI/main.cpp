#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("path_to_image.jpg"); // Use a valid image path here
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }
    cv::imshow("Image", img);
    cv::waitKey(0);
    return 0;
}