#include "utils.h"

#include <caffe/data_transformer.hpp>

#include <iostream>

const int ASCII_ESC = 27;
const int n_outputs = 15;
const float scale_sc = 100; // factor to multiply steering command with


// Show frames in leveldb
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  if(argc != 4) {
    LOG(ERROR) << "Usage: " << argv[0] << " input_db input_network input_weights";
    return 1;
  }

  // TODO currently this is hardcoded to use GPU 0 but it should try to
  // detect if there is a GPU and which one is best to use.
  const int gpu_idx = 0;
  caffe::Caffe::SetDevice(gpu_idx);
  caffe::Caffe::DeviceQuery();
  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  // open db
  std::string dbname(argv[1]);
  leveldb::Options options;
  options.error_if_exists = false;
  options.create_if_missing = false;
  options.max_open_files = 100;
  auto db = open_leveldb(dbname, options);
  auto shape = infer_shape(db);
  std::cout << "Inferred shape: " << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;

  // datum to be read
  caffe::Datum datum;

  // window to display
  unsigned int box_height = 60;
  IplImage* windowImg = cvCreateImage(cvSize(shape[2], shape[1] + box_height), IPL_DEPTH_8U, shape[0]);

  // Caffe model
  // First load prototxt describing the deployment setup
  caffe::NetParameter network_params;
  caffe::ReadProtoFromTextFile(argv[2], &network_params);
  caffe::Net<float> network(network_params);
  // Now load trained weights and copy as applicable
  caffe::NetParameter trained_network_params;
  caffe::ReadNetParamsFromBinaryFileOrDie(argv[3], &trained_network_params);
  network.CopyTrainedLayersFrom(trained_network_params);

  // input blob
  const std::vector<caffe::Blob<float>*>& input_blobs = network.input_blobs();
  CHECK(input_blobs.size() == 1) << "Expected a single input blob.";
  caffe::Blob<float>* input_blob = input_blobs[0];
  CHECK(input_blob->shape()[0] == 1) << "Input consists of a single frame.";
  CHECK(input_blob->shape()[1] == shape[0]) << "Frame size inconsistency.";
  CHECK(input_blob->shape()[2] == shape[1]) << "Frame size inconsistency.";
  CHECK(input_blob->shape()[3] == shape[2]) << "Frame size inconsistency.";

  // output blob
  const std::vector<caffe::Blob<float>*>& output_blobs = network.output_blobs();
  caffe::Blob<float>* output_blob = output_blobs[0];
  CHECK(output_blob->shape()[0] == 1) << "Output consists of prediction for a single frame";
  CHECK(output_blob->shape()[1] == n_outputs) << "Expected " << n_outputs << " outputs.";

  // Data transformer
  auto transformation_param = network.layers()[0]->layer_param().transform_param();
  caffe::DataTransformer<float> transformer(transformation_param, caffe::TEST);

  // iterate
  auto it = db->NewIterator(leveldb::ReadOptions());
  unsigned int count = 0;
  for(it->SeekToFirst(); it->Valid(); it->Next())
  {
    datum.ParseFromString(it->value().ToString());

    // clear window
    cvSet(windowImg, cvScalar(0,0,0));
    // draw current frame
    datum_to_ipl(datum, windowImg);

    // predict current frame
    transformer.Transform(datum, input_blob);
    network.Forward();
    // raw output data
    const float* output_data = output_blob->cpu_data();
    // visualize steering command
    float steering_command = output_data[n_outputs - 1];
    // TODO denormalize
    cvRectangle(windowImg,
                cvPoint(shape[2] / 2, shape[1]),
                cvPoint(shape[2] / 2 - scale_sc * steering_command, shape[1] + box_height / 2),
                cvScalar(237,99,157), -2);

    // use a constant name for the window otherwise a new window is created on each call
    cvShowImage(dbname.c_str(), windowImg);
    
    count += 1;
    auto key = cvWaitKey(20);
    if(key == ASCII_ESC) break;
  }
  std::cout << "Played a total of " << count << " keys." << std::endl;

  return 0;
}
