#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::string imagePath = "../data/cats_dogs/train/cats/Abyssinian_1.jpg";

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "can't read: " << imagePath << std::endl;
        return -1;
    }

    cv::imshow("show", image);
    cv::waitKey(0);
    return 0;
}

