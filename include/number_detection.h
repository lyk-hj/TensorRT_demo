#ifndef TENSORRT_NUMBER_DETECTION_H
#define TENSORRT_NUMBER_DETECTION_H

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>

//!
//! \brief check whether the operation on cuda is correct
//!
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if(ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while(0)

//!
//! \brief the specific operation of cuda deletion
//!
static auto StreamDeleter = [](cudaStream_t* pStream)
{
    if (pStream)
    {
        cudaStreamDestroy(*pStream);
        delete pStream;
    }
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

//!
//! \brief This class implements the number detection model by tensorRt
//!
class NumberDetection {
public:
    NumberDetection();

    bool build();

    bool infer(float* input, float* output);

    bool synLoadInfer();

    bool asynLoadInfer();

    float* processInput(cv::Mat src);//!< Reads the input and stores the result in a managed buffer to feed in neural network

//    int inputH;
//    int inputW;
//    int outputSize;
private:
    Logger gLogger;

    std::string onnx_path;

    const char *INPUT_BLOB_NAME = "input";
    const char *OUTPUT_BLOB_NAME = "output";
    const int inputH = 24;
    const int inputW = 24;
    const int inputC = 1;
    std::vector<std::string> classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8"};
    const int outputSize = classes.size();

    const int batchSize = 1;
//    const char* INPUT_BLOB_NAME = "input";
//    const char* OUTPUT_BLOB_NAME = "output";

    nvinfer1::Dims InuputDims;//!< The dimensions of the input to the network
    nvinfer1::Dims OutputDims;//!< The dimensions of the output to the network

    std::shared_ptr<nvinfer1::ICudaEngine> Engine;//!< The TensorRT engine used to run the network

    bool parserOnnxModel(nvinfer1::IBuilderConfig* config,
                         nvonnxparser::IParser* parser);//!< Parses an ONNX model for number detection and creates a TensorRT network

};

#endif //TENSORRT_NUMBER_DETECTION_H
