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

  string project_name = "resnet_v1_50_finetuned_4class_altered_model";

  parser->registerInput("images", DimsCHW(3, 244, 244));
  parser->registerOutput("resnet_v1_50/SpatialSqueeze");
  parser->parse("../weights/" + project_name + ".uff", *network, DataType::kFLOAT);

  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(1<<20);
  ICudaEngine *engine = builder->buildCudaEngine(*network);

  ofstream f;
  f.open("../weights/" + project_name + ".plan");
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
