//
// Created by hu_sh on 2022/5/7.
//

#include "v5lite.h"

int main(void){
    std::string config_file = "../config.yaml";
    std::string imgpath = "../0.jpg";

    V5lite V5lite(config_file);
    V5lite.LoadEngine();

    cv::Mat img = cv::imread(imgpath);
    std::vector<V5lite::DetectRes> res;
    for (int i = 0; i< 100; i++){
        V5lite.GetRst(img, res);
    }

    // draw
    for(const auto &rect : res)
    {
        char t[256];
        sprintf(t, "%.2f", rect.prob);
        std::string name = V5lite.coco_labels[rect.classes] + "-" + t;
        cv::putText(img, name, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - 5), cv::FONT_HERSHEY_COMPLEX, 0.7, V5lite.class_colors[rect.classes], 2);
        cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
        cv::rectangle(img, rst, V5lite.class_colors[rect.classes], 2, cv::LINE_8, 0);
    }
    cv::imwrite("out.jpg", img);

}