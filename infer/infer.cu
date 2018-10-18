#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string.h>

#include "utils.h"

using namespace std;
using namespace nvinfer1;

class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
    if (severity != Severity::kINFO)
      cout << msg << endl;
  }
} gLogger;

int main(int argc, char *argv[]) {
  // https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification/blob/master/examples/classify_image/classify_image.cu
  // load plan
  // https://github.com/NVIDIA-Jetson/tf_to_trt_image_classification/blob/master/examples/classify_image/classify_image.cu
  cout<<"started"<<endl;
  ifstream planFile(argv[1]);
  if(!planFile.is_open()) return 1;
  stringstream planBuffer;
  planBuffer << planFile.rdbuf();
  string plan = planBuffer.str();
  IRuntime *runtime = createInferRuntime(gLogger);
  ICudaEngine *engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);
  IExecutionContext *context = engine->createExecutionContext();

  int inputBindingIndex, outputBindingIndex;
  inputBindingIndex = engine->getBindingIndex("images");
  outputBindingIndex = engine->getBindingIndex("resnet_v1_50/SpatialSqueeze");
  Dims inputDims, outputDims;
  inputDims = engine->getBindingDimensions(inputBindingIndex);
  outputDims = engine->getBindingDimensions(outputBindingIndex);
  int inputWidth, inputHeight;
  inputHeight = inputDims.d[1];
  inputWidth = inputDims.d[2];
  cout<<inputHeight<<" "<<inputWidth<<endl;

  // 画像処理, preprocessing
  cv::Mat img = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB, 3);
  cv::resize(img, img, cv::Size(inputWidth, inputHeight));
  /* std::cout<<img.rows<<" "<<img.cols<<std::endl; */
  /* cv::imshow("", img); */
  /* cv::waitKey(0); */
  size_t numInput, numOutput;
  numInput = numTensorElements(inputDims);
  numOutput = numTensorElements(outputDims);
  float *inputDataHost, *outputDataHost;
  float *inputDataDevice, *outputDataDevice;

  // GPU-CPU共有メモリを使うかどうかのフラグ
  bool useMappedMemory = (strcmp(argv[3], "map")==0) ? true : false;
  if (useMappedMemory) {
    cudaHostAlloc(&inputDataHost, numInput * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc(&outputDataHost, numOutput * sizeof(float), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&inputDataDevice, inputDataHost, 0);
    cudaHostGetDevicePointer(&outputDataDevice, outputDataHost, 0);
  }else {
    inputDataHost = (float*) malloc(numInput * sizeof(float));
    outputDataHost = (float*) malloc(numOutput * sizeof(float));
    cudaMalloc(&inputDataDevice, numInput * sizeof(float));
    cudaMalloc(&outputDataDevice, numOutput * sizeof(float));
  }
  void *bindings[2];
  bindings[inputBindingIndex] = (void*) inputDataDevice;
  bindings[outputBindingIndex] = (void*) outputDataDevice;

  for(int t=0; t<10; t++) {
    auto t0 = chrono::steady_clock::now();
    cvImageToTensor(img, inputDataHost, inputDims);
    preprocessVgg(inputDataHost, inputDims);
    auto t1 = chrono::steady_clock::now();
    if (useMappedMemory) {
      // 実行
      context->execute(1, bindings);
      /* cout<<outputDataHost[0]<<endl; */
    } else {
      // 画像のコピー
      cudaMemcpy(inputDataDevice, inputDataHost, numInput * sizeof(float), cudaMemcpyHostToDevice);
      // 実行
      context->execute(1, bindings);
      // 結果のコピー
      cudaMemcpy(outputDataHost, outputDataDevice, numOutput * sizeof(float), cudaMemcpyDeviceToHost);
    }
    auto t2 = chrono::steady_clock::now();
    cout<<chrono::duration_cast<chrono::milliseconds>(t1 - t0).count()<<"ms "<<chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()<<"ms"<<endl;
    for(int i=0; i<4; i++) {
      cout<<outputDataHost[i]<<" ";
    }
    cout<<endl;
  }

  /*
  for(int i=0; i<numOutput; i++){
    cout<<i<<":"<<outputDataHost[i]<<endl;
  }
  */

  runtime->destroy();
  engine->destroy();
  context->destroy();
  if(useMappedMemory) {
    cudaFreeHost(inputDataHost);
    cudaFreeHost(outputDataHost);
  }else{
    free(inputDataHost);
    free(outputDataHost);
  }
  cudaFree(inputDataDevice);
  cudaFree(outputDataDevice);

  return 0;
}
