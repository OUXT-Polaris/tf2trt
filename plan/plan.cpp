#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <NvInfer.h>
#include <NvUffParser.h>

using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;


// コピペエリア
class Logger : public ILogger
{
  void log(Severity severity, const char * msg) override
  {
    cout << msg << endl;
  }
} gLogger;


int main(int argc, char *argv[]) {
  cout<<2<<endl;

  IBuilder *builder = createInferBuilder(gLogger);
  INetworkDefinition *network = builder->createNetwork();
  IUffParser *parser = createUffParser();

  parser->registerInput("images", DimsCHW(3, 224, 224));
  parser->registerOutput("resnet_v1_50/SpatialSqueeze");
  parser->parse("resnet_v1_50_finetuned_4class_altered_model.uff", *network, DataType::kFLOAT);  // or, kHALF

  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(1<<20);
  ICudaEngine *engine = builder->buildCudaEngine(*network);

  ofstream f;
  f.open("resnet_v1_50_finetuned_4class_altered_model_float.plan");
  IHostMemory *serializedEngine = engine->serialize();
  f.write((char *)serializedEngine->data(), serializedEngine->size());
  f.close();

  builder->destroy();
  parser->destroy();
  network->destroy();
  engine->destroy();
  serializedEngine->destroy();

  return 0;
}
