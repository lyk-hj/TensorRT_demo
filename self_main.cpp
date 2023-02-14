#include <fstream>
#include "number_detection.h"

using namespace nvinfer1;

nvinfer1::IBuilder *attempt() {
    nvinfer1::IBuilder *builder;
    int a = 5 + 6;
    printf("the value of a: %d\n", a);
    return builder;
}

int main() {
    try{
        //!< preset parameters
        const char *INPUT_BLOB_NAME = "input";
        const char *OUTPUT_BLOB_NAME = "output";
        const int inputH = 30;
        const int inputW = 20;
        const int inputC = 1;
        const int outputSize = 6;
        std::string classes[]{"0", "1", "2", "3", "4", "6"};
        //        cv::FileStorage fs("../other/trt_params.yaml", cv::FileStorage::READ);
        std::vector<std::string> sample_paths = {"../sample/2.png",
                                                 "../sample/3.png",
                                                 "../sample/2.png",
                                                 "../sample/4.png"};
        //        std::cout << sample_path << std::endl;
        //        fs.release();

        //!< load model and create context
        std::ifstream file("../model/2023_2_1_hj_num_2_fp16.trt", std::ios::binary);
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        char *trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
        IRuntime *runtime = createInferRuntime(gLogger);
        ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
        Dims batch = engine->getBindingDimensions(0);
        std::cout<<batch.d[2]<<std::endl;
        IExecutionContext *context = engine->createExecutionContext();
        delete[] trtModelStream;
        const size_t batchSize = sample_paths.size();

        std::vector<void*> buffers(engine->getNbBindings());
        std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream(new cudaStream_t, StreamDeleter);
        CHECK(cudaStreamCreate(stream.get()));

        const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * inputC * inputH * inputW * sizeof(float)));

        float *buffInputIdx = (float*)buffers[inputIndex];
        for(int i=0;i<batchSize;i++)
        {
            //!< read image
            cv::Mat src = cv::imread(sample_paths[i]);
            cv::resize(src,src,cv::Size(inputW,inputH));
            cv::cvtColor(src,src,cv::COLOR_BGR2GRAY);
            cv::threshold(src,src,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
            src.convertTo(src,CV_32FC1);
            src /= 255.0f;

//            assert(engine->getNbBindings() == 2);
            //!< memcpy data to device
            std::shared_ptr<float> input = std::shared_ptr<float>(new float[inputC*inputH*inputW]);
            cv::Mat linkData(src.size(),src.type(),input.get());
            src.copyTo(linkData);
            CHECK(cudaMemcpyAsync(
                    (float*)buffers[inputIndex]+inputC*inputH*inputW*i, input.get(), inputC * inputH * inputW * sizeof(float),
                    cudaMemcpyHostToDevice, *stream
            ));
//            buffInputIdx+=inputC*inputH*inputW;
        }

//        if(context->getEngine().hasImplicitBatchDimension())printf("context has implicit\n");
//        else printf("context hasn't implicit\n");

        //!< inference
        auto start = std::chrono::high_resolution_clock::now();
        float* output = new float[outputSize*batchSize];
        const int& outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputSize * sizeof(float)));
        context->setInputShape(INPUT_BLOB_NAME,Dims4(batchSize,1,30,20));
        context->enqueueV2(buffers.data(),*stream, nullptr);
        CHECK(cudaMemcpyAsync(
                output, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost, *stream
                ));
        cudaStreamSynchronize(*stream);

        //Release buffers
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));

        for(int j=0;j<batchSize;j++)
        {
            std::cout << "-----Output------" <<j <<"\n\n";
            for (int i = 0; i < outputSize; i++) {
                std::cout << std::left << std::setw(10) << output[i+j*outputSize] << "    ";
                std::cout << std::left <<std::setw(8) << std::fixed << classes[i]<< std::endl;
            }
        }
        delete context;
        delete engine;
        delete runtime;
        output = nullptr;
        delete output;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end-start).count();
        printf("Inference duration: %lf\n", duration);

    }catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}