#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "box_utils.h"

#define CHECK(status)                                    \
  do                                                     \
  {                                                      \
    auto ret = (status);                                 \
    if (ret != 0)                                        \
    {                                                    \
      std::cerr << "Cuda failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0 // GPU id
#define NMS_THRESH 0.04
#define BBOX_CONF_THRESH 0.5

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int OUTPUT_SIZE_CNF = ssd::NUM_CLASSES * ssd::NUM_DETECTIONS;
static const int OUTPUT_SIZE_BX = ssd::LOCATIONS * ssd::NUM_DETECTIONS;
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME_CNF = "confidences";
const char *OUTPUT_BLOB_NAME_BX = "locations";
static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
  std::cout << "Loading weights: " << file << std::endl;
  std::map<std::string, Weights> weightMap;

  // Open weights file
  std::ifstream input(file);
  assert(input.is_open() && "Unable to load weight file.");

  // Read number of weight blobs
  int32_t count;
  input >> count;
  assert(count > 0 && "Invalid weight map file.");
  std::cout << "Total weights: " << count << std::endl;

  while (count--)
  {
    Weights wt{DataType::kFLOAT, nullptr, 0};
    uint32_t size;

    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> size;
    wt.type = DataType::kFLOAT;

    // Load blob
    uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
    for (uint32_t x = 0, y = size; x < y; ++x)
    {
      input >> std::hex >> val[x];
    }
    wt.values = val;
    wt.count = size;
    weightMap[name] = wt;
  }

  return weightMap;
}

IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps)
{
  float *gamma = (float *)weightMap[lname + ".weight"].values;
  float *beta = (float *)weightMap[lname + ".bias"].values;
  float *mean = (float *)weightMap[lname + ".running_mean"].values;
  float *var = (float *)weightMap[lname + ".running_var"].values;
  int len = weightMap[lname + ".running_var"].count;

  float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++)
  {
    scval[i] = gamma[i] / sqrt(var[i] + eps);
  }
  Weights scale{DataType::kFLOAT, scval, len};

  float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++)
  {
    shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
  }
  Weights shift{DataType::kFLOAT, shval, len};

  float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++)
  {
    pval[i] = 1.0;
  }
  Weights power{DataType::kFLOAT, pval, len};

  weightMap[lname + ".scale"] = scale;
  weightMap[lname + ".shift"] = shift;
  weightMap[lname + ".power"] = power;
  IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
  assert(scale_1);

  return scale_1;
}

IActivationLayer *convBn(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int ksize, int s, int p, std::string lname)
{
  Weights emptywts{DataType::kFLOAT, nullptr, 0};
  IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[lname + "0.weight"], emptywts);
  assert(conv1);
  conv1->setStride(DimsHW{s, s});
  conv1->setPadding(DimsHW{p, p});

  IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

  IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
  assert(relu1);

  return relu1;
}

IActivationLayer *convDw(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname,
                         int inch, int outch, int s)
{
  Weights emptywts{DataType::kFLOAT, nullptr, 0};

  IConvolutionLayer *conv1 = network->addConvolutionNd(input, inch, DimsHW{3, 3}, weightMap[lname + "0.weight"], emptywts);
  assert(conv1);
  conv1->setStride(DimsHW{s, s});
  conv1->setPadding(DimsHW{1, 1});
  conv1->setNbGroups(inch);
  IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);
  IActivationLayer *relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
  assert(relu1);

  IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "3.weight"], emptywts);
  assert(conv2);
  conv2->setStride(DimsHW{1, 1});
  conv2->setPadding(DimsHW{0, 0});
  IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "4", 1e-5);
  IActivationLayer *relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
  assert(relu2);

  return relu2;
}

IActivationLayer *extras(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname,
                         int outch1, int outch2)
{

  IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch1, DimsHW{1, 1}, weightMap[lname + "0.weight"], weightMap[lname + "0.bias"]);
  assert(conv1);
  conv1->setStride(DimsHW{1, 1});
  conv1->setPadding(DimsHW{0, 0});
  IActivationLayer *relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  assert(relu1);

  IConvolutionLayer *conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch2, DimsHW{3, 3}, weightMap[lname + "2.weight"], weightMap[lname + "2.bias"]);
  assert(conv2);
  conv2->setStride(DimsHW{2, 2});
  conv2->setPadding(DimsHW{1, 1});
  IActivationLayer *relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
  assert(relu2);

  return relu2;
}

IShuffleLayer *reshapeHeader(INetworkDefinition *network, ITensor &input, int size)
{
  // Permutation and reshape the tensor for classification
  IShuffleLayer *re = network->addShuffle(input);
  assert(re);

  Permutation p1 = {{2, 1, 0}};
  re->setFirstTranspose(p1);
  re->setReshapeDimensions(Dims2(-1, size));

  return re;
}

ILayer *detectionHeader(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, int outch, int outsize)
{

  IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + ".weight"], weightMap[lname + ".bias"]);
  conv1->setStride(DimsHW{1, 1});
  conv1->setPadding(DimsHW{1, 1});
  assert(conv1);

  IShuffleLayer *re1 = reshapeHeader(network, *conv1->getOutput(0), outsize);
  return re1;
}

// Create the engine using only the API and not any parser.
ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
{
  INetworkDefinition *network = builder->createNetworkV2(0U);

  // Create input tensor with name INPUT_BLOB_NAME
  ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ssd::INPUT_C, ssd::INPUT_H, ssd::INPUT_W});
  assert(data);

  std::map<std::string, Weights> weightMap = loadWeights("../mobilenet-v1-ssd.wts");

  IActivationLayer *relu0 = convBn(network, weightMap, *data, 32, 3, 2, 1, "base_net.0.");
  IActivationLayer *relu1 = convDw(network, weightMap, *relu0->getOutput(0), "base_net.1.", 32, 64, 1);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.2.", 64, 128, 2);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.3.", 128, 128, 1);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.4.", 128, 256, 2);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.5.", 256, 256, 1);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.6.", 256, 512, 2);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.7.", 512, 512, 1);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.8.", 512, 512, 1);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.9.", 512, 512, 1);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.10.", 512, 512, 1);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.11.", 512, 512, 1);

  ILayer *ch0 = detectionHeader(network, weightMap, *relu1->getOutput(0), "classification_headers.0", ssd::NUM_CLASSES * 6, ssd::NUM_CLASSES);
  ILayer *rh0 = detectionHeader(network, weightMap, *relu1->getOutput(0), "regression_headers.0", ssd::LOCATIONS * 6, ssd::LOCATIONS);

  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.12.", 512, 1024, 2);
  relu1 = convDw(network, weightMap, *relu1->getOutput(0), "base_net.13.", 1024, 1024, 1);

  ILayer *ch1 = detectionHeader(network, weightMap, *relu1->getOutput(0), "classification_headers.1", ssd::NUM_CLASSES * 6, ssd::NUM_CLASSES);
  ILayer *rh1 = detectionHeader(network, weightMap, *relu1->getOutput(0), "regression_headers.1", ssd::LOCATIONS * 6, ssd::LOCATIONS);

  IActivationLayer *relu2 = extras(network, weightMap, *relu1->getOutput(0), "extras.0.", 256, 512);

  ILayer *ch2 = detectionHeader(network, weightMap, *relu2->getOutput(0), "classification_headers.2", ssd::NUM_CLASSES * 6, ssd::NUM_CLASSES);
  ILayer *rh2 = detectionHeader(network, weightMap, *relu2->getOutput(0), "regression_headers.2", ssd::LOCATIONS * 6, ssd::LOCATIONS);

  relu2 = extras(network, weightMap, *relu2->getOutput(0), "extras.1.", 128, 256);

  ILayer *ch3 = detectionHeader(network, weightMap, *relu2->getOutput(0), "classification_headers.3", ssd::NUM_CLASSES * 6, ssd::NUM_CLASSES);
  ILayer *rh3 = detectionHeader(network, weightMap, *relu2->getOutput(0), "regression_headers.3", ssd::LOCATIONS * 6, ssd::LOCATIONS);

  relu2 = extras(network, weightMap, *relu2->getOutput(0), "extras.2.", 128, 256);

  ILayer *ch4 = detectionHeader(network, weightMap, *relu2->getOutput(0), "classification_headers.4", ssd::NUM_CLASSES * 6, ssd::NUM_CLASSES);
  ILayer *rh4 = detectionHeader(network, weightMap, *relu2->getOutput(0), "regression_headers.4", ssd::LOCATIONS * 6, ssd::LOCATIONS);

  relu2 = extras(network, weightMap, *relu2->getOutput(0), "extras.3.", 128, 256);

  ILayer *ch5 = detectionHeader(network, weightMap, *relu2->getOutput(0), "classification_headers.5", ssd::NUM_CLASSES * 6, ssd::NUM_CLASSES);
  ILayer *rh5 = detectionHeader(network, weightMap, *relu2->getOutput(0), "regression_headers.5", ssd::LOCATIONS * 6, ssd::LOCATIONS);

  ITensor *chTensor[] = {ch0->getOutput(0), ch1->getOutput(0), ch2->getOutput(0), ch3->getOutput(0), ch4->getOutput(0), ch5->getOutput(0)};
  IConcatenationLayer *chConcat = network->addConcatenation(chTensor, 6);
  ISoftMaxLayer *conf = network->addSoftMax(*chConcat->getOutput(0));
  conf->setAxes(1 << 1);

  ITensor *rhTensor[] = {rh0->getOutput(0), rh1->getOutput(0), rh2->getOutput(0), rh3->getOutput(0), rh4->getOutput(0), rh5->getOutput(0)};
  IConcatenationLayer *rhConcat = network->addConcatenation(rhTensor, 6);

  std::cout << std::endl
            << "Total layers in the network: " << network->getNbLayers() << std::endl;

  // Mobilenet classifier code
  std::cout << std::endl
            << "Building engine ..." << std::endl;
  conf->getOutput(0)->setName(OUTPUT_BLOB_NAME_CNF);
  network->markOutput(*conf->getOutput(0));

  rhConcat->getOutput(0)->setName(OUTPUT_BLOB_NAME_BX);
  network->markOutput(*rhConcat->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#ifdef USE_FP16
  config->setFlag(BuilderFlag::kFP16);
#endif
  ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "build out" << std::endl;

  // Don't need the network any more
  network->destroy();

  // Release host memory
  for (auto &mem : weightMap)
  {
    free((void *)(mem.second.values));
  }

  return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream)
{
  // Create builder
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an engine
  ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
  assert(engine != nullptr);

  // Serialize the engine
  (*modelStream) = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();
}

void doInference(IExecutionContext &context, float *input, float *output_cnf, float *output_bx, int batchSize)
{
  const ICudaEngine &engine = context.getEngine();

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 3);
  void *buffers[3];

  // In order to bind the buffers, we need to know the names of the input and output tensors.
  // Note that indices are guaranteed to be less than IEngine::getNbBindings()
  const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
  const int outputIndexCnf = engine.getBindingIndex(OUTPUT_BLOB_NAME_CNF);
  const int outputIndexBx = engine.getBindingIndex(OUTPUT_BLOB_NAME_BX);

  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex], batchSize * ssd::INPUT_C * ssd::INPUT_H * ssd::INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndexCnf], batchSize * OUTPUT_SIZE_CNF * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndexBx], batchSize * OUTPUT_SIZE_BX * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * ssd::INPUT_C * ssd::INPUT_H * ssd::INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
  context.enqueue(batchSize, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output_cnf, buffers[outputIndexCnf], batchSize * OUTPUT_SIZE_CNF * sizeof(float), cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(output_bx, buffers[outputIndexBx], batchSize * OUTPUT_SIZE_BX * sizeof(float), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndexCnf]));
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names)
{
  DIR *p_dir = opendir(p_dir_name);
  if (p_dir == nullptr)
  {
    return -1;
  }

  struct dirent *p_file = nullptr;
  while ((p_file = readdir(p_dir)) != nullptr)
  {
    if (strcmp(p_file->d_name, ".") != 0 &&
        strcmp(p_file->d_name, "..") != 0)
    {
      std::string cur_file_name(p_file->d_name);
      file_names.push_back(cur_file_name);
    }
  }

  closedir(p_dir);
  return 0;
}

int main(int argc, char **argv)
{
  cudaSetDevice(DEVICE);
  // create a model using the API directly and serialize it to a stream
  char *trtModelStream{nullptr};
  size_t size{0};

  if (argc == 2 && std::string(argv[1]) == "-s")
  {
    IHostMemory *modelStream{nullptr};
    APIToModel(1, &modelStream);
    assert(modelStream != nullptr);

    std::ofstream p("mobilenet-v1-ssd.engine");
    if (!p)
    {
      std::cerr << "could not open plan output file" << std::endl;
      return -1;
    }
    p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    std::cout << "Saved as mobilenet-v1-ssd.engine" << std::endl;
    return 0;
  }
  else if (argc == 3 && std::string(argv[1]) == "-d")
  {
    std::ifstream file("mobilenet-v1-ssd.engine", std::ios::binary);
    if (file.good())
    {
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream = new char[size];
      assert(trtModelStream);
      file.read(trtModelStream, size);
      file.close();
      std::cout << "Engine file read successful" << std::endl;
    }
  }
  else
  {
    std::cerr << "arguments not right!" << std::endl;
    std::cerr << "./mobilenet -s             // serialize model to plan file" << std::endl;
    std::cerr << "./mobilenet -d ../samples  // deserialize plan file and run inference" << std::endl;
    return -1;
  }

  std::vector<std::string> file_names;
  if (read_files_in_dir(argv[2], file_names) < 0)
  {
    std::cout << "read_files_in_dir failed." << std::endl;
    return -1;
  }

  float data[ssd::INPUT_C * ssd::INPUT_H * ssd::INPUT_W];

  IRuntime *runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
  assert(engine != nullptr);
  IExecutionContext *context = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trtModelStream;

  for (int f = 0; f < (int)file_names.size(); f++)
  {
    cv::Mat img = cv::imread(std::string(argv[2]) + "/" + file_names[f]);
    if (img.empty())
      continue;

    cv::Mat pr_img = img.clone();
    cv::resize(pr_img, pr_img, cv::Size(ssd::INPUT_H, ssd::INPUT_W));

    for (int i = 0; i < ssd::INPUT_H * ssd::INPUT_W; i++)
    {
      data[i] = (pr_img.at<cv::Vec3b>(i)[2] - 127) / 128.0;
      data[i + ssd::INPUT_H * ssd::INPUT_W] = (pr_img.at<cv::Vec3b>(i)[1] - 127) / 128.0;
      data[i + 2 * ssd::INPUT_H * ssd::INPUT_W] = (pr_img.at<cv::Vec3b>(i)[0] - 127) / 128.0;
    }

    // Run inference
    float prob[OUTPUT_SIZE_CNF], locations[OUTPUT_SIZE_BX];
    auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, locations, 1);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<ssd::Detection> dt = post_process_output(prob, locations, BBOX_CONF_THRESH, NMS_THRESH);
    // std::cout << "Num detections " << dt.size() << std::endl;
    for (size_t i = 0; i < dt.size(); i++)
    {
      float xmin = dt[i].bbox[0];
      float ymin = dt[i].bbox[1];
      float xmax = dt[i].bbox[2];
      float ymax = dt[i].bbox[3];
      int x1 = static_cast<int>(xmin * img.cols);
      int y1 = static_cast<int>(ymin * img.rows);
      int x2 = static_cast<int>(xmax * img.cols);
      int y2 = static_cast<int>(ymax * img.rows);
      cv::rectangle(img, cv::Rect2f(cv::Point(x1, y1), cv::Point(x2, y2)), cv::Scalar(255, 0, 255), 1);
      cv::putText(img, std::to_string((int)dt[i].class_id), cv::Point(x1, y1 - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
    cv::imwrite("_" + file_names[f], img);
  }

  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  return 0;
}
