#include "number_detection.h"

using namespace nvinfer1;
using namespace std;
using namespace cv;


NumberDetection::NumberDetection()
{
    FileStorage fs("../other/trt_params.yaml",FileStorage::READ);
    onnx_path = (string)fs["onnx_path"];
    fs.release();
}


bool NumberDetection::build()
{
    IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    if(!builder)
    {
        printf("build failed!!!");
        return false;
    }

    INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!network)
    {
        printf("network failed!!!");
        return false;
    }

    IBuilderConfig* config = builder->createBuilderConfig();
    if (!config)
    {
        printf("config failed!!!");
        return false;
    }

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network,gLogger);
    if (!parser)
    {
        printf("parser failed!!!");
        return false;
    }

    bool constructed = parserOnnxModel(config,parser);
    if (!constructed)
    {
        printf("constructed failed!!!");
        return false;
    }

    //creat CUDA stream
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> trtStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(trtStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        printf("trt cuda stream creat failed!!!");
        return false;
    }
    config->setProfileStream(*trtStream);

    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        printf("plan failed!!!");
        return false;
    }
//
//    std::unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
//    if (!runtime)
//    {
//        printf("runtime failed!!!");
//        return false;
//    }
//
//    Engine = std::shared_ptr<ICudaEngine>(
//            runtime->deserializeCudaEngine(plan->data(), plan->size())
//            );
//    if (!Engine)
//    {
//        printf("Engine failed!!!");
//        return false;
//    }

//    gLogger.log(ILogger::Severity::kINFO,
//               to_string(network->getNbInputs()).c_str());//should be 1
//    InuputDims = network->getInput(0)->getDimensions();
//    inputH = InuputDims.d[2];
//    inputW = InuputDims.d[3];
//    gLogger.log(ILogger::Severity::kINFO,
//               to_string(InuputDims.d[0]).c_str());//should be -1
//
////    gLogger.log(ILogger::Severity::kINFO,
////               to_string(network->getNbOutputs()).c_str());//should be 1
//    OutputDims = network->getOutput(0)->getDimensions();
//    outputSize = OutputDims.d[1];
//    gLogger.log(ILogger::Severity::kINFO,
//               to_string(OutputDims.d[0]).c_str());//should be -1
    delete parser;
    delete config;
    delete network;
    delete builder;
    return true;


}

bool NumberDetection::infer(float *input, float* output)
{
    IExecutionContext* context = Engine->createExecutionContext();
    if (!context)
    {
        printf("context failed!!!");
        return false;
    }

    assert(Engine->getNbBindings() == 2);
    void* buffers[2];

    const int inputIndex = Engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = Engine->getBindingIndex(OUTPUT_BLOB_NAME);

    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 1 * inputH * inputW * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize*outputSize*sizeof(float)));

    unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream(new cudaStream_t, StreamDeleter);
    CHECK(cudaStreamCreate(stream.get()));

    CHECK(cudaMemcpyAsync(
            buffers[inputIndex], input, batchSize*1*inputH*inputW*sizeof(float),cudaMemcpyHostToDevice,*stream
            ));
    context->enqueueV2(buffers, *stream, nullptr);
    CHECK(cudaMemcpyAsync(
            output, buffers[outputIndex], batchSize*outputSize*sizeof(float),cudaMemcpyDeviceToHost,*stream
            ));
    cudaStreamSynchronize(*stream);

    //Release buffers
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    return true;
}

bool NumberDetection::parserOnnxModel(IBuilderConfig* config,
                                      nvonnxparser::IParser* parser)
{
    auto parsed = parser->parseFromFile(onnx_path.c_str(),
                                        static_cast<int>(ILogger::Severity::kWARNING));

    if (!parsed)
    {
        return false;
    }

    //set the FP16 mode
    config->setFlag(BuilderFlag::kFP16);
    //set that when the layer cannot implement in DLA, it will implement in GPU
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    //specify default running device
    config->setDefaultDeviceType(DeviceType::kDLA);
    //specify which DLA core to use via indexing
    config->setDLACore(0);

    return true;

}

float* NumberDetection::processInput(Mat src)
{
    Mat input;
    cvtColor(src,src,COLOR_BGR2GRAY);
    threshold(src,src,0,255,THRESH_BINARY || THRESH_OTSU);
    src.convertTo(input,CV_32F);
    input /= 255.0;
    return input.ptr<float>(0);
}

bool NumberDetection::synLoadInfer()
{
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
                                                 "../sample/6_7.png",
                                                 "../sample/4.png"};
        //        std::cout << sample_path << std::endl;
        //        fs.release();

        //!< load model and create context
        std::ifstream file("../model/2023_2_1_hj_num_2.trt", std::ios::binary);
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        char *trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
        IRuntime *runtime = createInferRuntime(gLogger);
        ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
        //        Dims batch = engine->getBindingDimensions(0);
        //        std::cout<<batch.d[2]<<std::endl;
        IExecutionContext *context = engine->createExecutionContext();
        delete[] trtModelStream;
        const size_t batchSize = sample_paths.size();

        std::vector<void*> buffers(engine->getNbBindings());
        //        std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream(new cudaStream_t, StreamDeleter);
        //        CHECK(cudaStreamCreate(stream.get()));

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
            CHECK(cudaMemcpy(
                    (float*)buffers[inputIndex]+inputC*inputH*inputW*i, input.get(), inputC * inputH * inputW * sizeof(float),
                    cudaMemcpyHostToDevice
                    ));
            //            buffInputIdx+=inputC*inputH*inputW;
        }

        if(context->getEngine().hasImplicitBatchDimension())printf("context has implicit\n");
        else printf("context hasn't implicit\n");

        //!< inference
        auto start = std::chrono::high_resolution_clock::now();
        float* output = new float[outputSize*batchSize];
        const int& outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
        //        std::cout<<inputIndex<<std::endl;
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputSize * sizeof(float)));
        context->setInputShape(INPUT_BLOB_NAME,Dims4(batchSize,1,30,20));
        std::cout<<context->getBindingDimensions(0).d[2]<<std::endl;
        context->executeV2(buffers.data());
        CHECK(cudaMemcpy(
                output, buffers[outputIndex], batchSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost
                ));
        //        cudaStreamSynchronize(*stream);

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
        return true;
    }catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return false;
    }
}

