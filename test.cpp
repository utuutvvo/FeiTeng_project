#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

struct divide_result{
    Mat capture;
    Point2f centernum;
};

Mat frame;

void First_method2Deal(Mat scr)
 {
    divide_result result;
    Mat foreGround,fgMaskMOG2, hsv, mask,kernel,final;
    Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();

    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    pMOG2->apply(frame, fgMaskMOG2);
    cvtColor(scr,hsv,COLOR_BGR2HSV);
    Scalar lower_green(35,60,60);
    Scalar upper_green(80,255,255);//颜色追踪范围

    inRange(hsv, lower_green, upper_green, mask);
    foreGround = Scalar::all(0);
    mask.copyTo(foreGround, fgMaskMOG2);
    GaussianBlur(foreGround,foreGround,Size(5,5),0);//滤波

    morphologyEx(foreGround, foreGround, MORPH_OPEN, kernel, Point(-1, -1), 1);
    morphologyEx(foreGround, foreGround, MORPH_CLOSE, kernel, Point(-1, -1), 2);//形态学操作

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(foreGround, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double max_area = 0;
    int max_area_contour_index = -1;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > max_area && area > 300) 
        {
            max_area = area;
            max_area_contour_index = i;
        }
    }

    if (max_area_contour_index != -1) 
    {
        Rect bounding_rect = cv::boundingRect(contours[max_area_contour_index]);
        rectangle(frame, bounding_rect, Scalar(0, 255, 0), 2);
        string dimensions = "Width: " + to_string(bounding_rect.width) + ", Height: " + to_string(bounding_rect.height);
        putText(frame, dimensions, Point(bounding_rect.x, bounding_rect.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }

    if (max_area_contour_index != -1) {
        RotatedRect rotated_rect = minAreaRect(contours[max_area_contour_index]);
        Point2f vertices[4];
        Point2f center = rotated_rect.center;
        rotated_rect.points(vertices);

        vector<vector<Point>> contour;
        vector<Point> pts;
        for (int i = 0; i < 4; i++) {
            pts.push_back(vertices[i]);
        }
        contour.push_back(pts);
        Size2f rect_size = rotated_rect.size;
        double Width, Height;
        Width = rect_size.width;
        Height = rect_size.height;
        int W = (int)Width;
        int H = (int)Height;
        drawContours(frame, contour, -1, Scalar(255, 0, 0), 2);
        string dimensions = "Width:" + to_string(W) + ",Height:" + to_string(H) + ",center_X:" + to_string(center.x) + ",center_Y:" + std::to_string(center.y);
        putText(frame, dimensions, Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

    }

 }

int main()
{
    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cerr<<"无法打开摄像头"<<endl;
        return -1;
    }

    namedWindow("网络摄像头", WINDOW_NORMAL);

    while(true)
    {
        cap>>frame;
        
        if(frame.empty())
        {
            cerr<<"无法捕获图片"<<endl;
            continue;
        }

        //此处为图像处理部分
        First_method2Deal(frame);

        imshow("网络摄像头",frame);

        char key = waitKey(10);
        if(key == 'q')
        {
            break;
        }

    }
    cap.release();
    destroyAllWindows();

    return 0;
}