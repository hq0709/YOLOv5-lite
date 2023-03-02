#ifndef V5lite_TRT_V5lite_H
#define V5lite_TRT_V5lite_H

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

class V5lite
{


public:

    struct DetectRes{
        int classes;
        float x;
        float y;
        float w;
        float h;
        float prob;
    };

    V5lite(const std::string &config_file);
    ~V5lite();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);
    bool GetRst(cv::Mat &src_img, std::vector<V5lite::DetectRes> & res);

    std::vector<cv::Scalar> class_colors;
    std::map<int, std::string> coco_labels;

private:
    void EngineInference(const std::vector<std::string> &image_list, const int &outSize,void **buffers,
                         const std::vector<int64_t> &bufferSize, cudaStream_t stream);
    std::vector<float> prepareImage(std::vector<cv::Mat> & vec_img);
    std::vector<std::vector<DetectRes>> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
    void NmsDetect(std::vector <DetectRes> &detections);
    float IOUCalculate(const DetectRes &det_a, const DetectRes &det_b);
    std::string onnx_file;
    std::string engine_file;
    std::string labels_file;

    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int CATEGORY;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    float obj_threshold;
    float nms_threshold;
    std::vector<int> strides;
    std::vector<int> num_anchors;
    std::vector<std::vector<int>> anchors;
    std::vector<std::vector<int>> grids;





    void *buffers[2];
    std::vector<int64_t> bufferSize;
    cudaStream_t stream;

    std::vector<float> prepareimg(cv::Mat &img);

};

#endif 
