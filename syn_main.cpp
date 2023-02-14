#include "number_detection.h"

using namespace nvinfer1;

nvinfer1::IBuilder *attempt() {
    nvinfer1::IBuilder *builder;
    int a = 5 + 6;
    printf("the value of a: %d\n", a);
    return builder;
}

int main() {
    NumberDetection numberDetection;

//    if (!numberDetection.build())
//    {
//        printf("build failed!!!");
//    }
//    if(!numberDetection.infer())
    numberDetection.synLoadInfer();

    return 0;
}