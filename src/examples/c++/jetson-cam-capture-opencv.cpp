/*
Compile with:

gcc -std=c++11 Camera_Grab.cpp -o picture_grabber -L/usr/lib -lstdc++ -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs

Requires recompiling OpenCV with gstreamer plug in on. See: https://github.com/jetsonhacks/buildOpenCVTX2

Credit to Peter Moran for base code.
http://petermoran.org/csi-cameras-on-tx2/
*/


#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

std::string get_tegra_pipeline(int width, int height, int fps) {
    return "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)I420, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main() {
    // Options
    int WIDTH = 500;
    int HEIGHT = 500;
    int FPS = 30;

    // Directory name
    string set_dir = "Test";
    // Image base name
    string name_type = "test";

    int count = 0;

    // Define the gstream pipeline
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);
    std::cout << "Using pipeline: \n\t" << pipeline << "\n";

    // Create OpenCV capture object, ensure it works.
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cout << "Connection failed";
        return -1;
    }

    // View video
    cv::Mat frame;
    while (1) {
        cap >> frame;  // Get a new frame from camera

        // Display frame
        imshow("Display window", frame);

        // Press the esc to take picture or hold it down to really take a lot!
        if (cv::waitKey(1) % 256 == 27){

            string string_num = to_string(count);

            cout << "Now saving: " << string_num << endl;

            string save_location = "./" + set_dir + "/" + name_type + "_" + string_num + ".png";

            cout << "Save location: " << save_location << endl;

            imwrite(save_location, frame );

            count+=1;

        }

        cv::waitKey(1);
    }
}
