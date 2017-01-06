#include "utils.h"

#include <caffe/data_transformer.hpp>

#include <gflags/gflags.h>

#include <iostream>

const int n_outputs = 15;

std::map<char, std::string> bindings{
  {'q', "quit"},
  {'l', "fast_forward"},
  {'h', "slow_down"},
  {'+', "jump_hundred_forward"},
  {'-', "jump_hundred_back"}};
    

DEFINE_string(dbname, "torcs_visualization_input", "Database containing frames to be visualized.");
DEFINE_string(dbname_ground_truth, "torcs_visualization_target", "Database containing ground truth.");
DEFINE_string(network_prototxt, "network_deploy.prototxt", "Prototxt describing network architecture.");
DEFINE_string(network_caffemodel, "network_weights.caffemodel", "Caffemodel with the weights to use for network.");
DEFINE_string(normalization_protobinary, "torcs_train_normalization.binaryproto", "Protobinary containing Blob with normalization parameters.");

DEFINE_int32(start_frame, 0, "Frame to start with.");

// Show frames in leveldb
int main(int argc, char** argv) {
  gflags::SetUsageMessage("Visualize a database containing frames from TORCS and the predictions of a specified network for the steering command.");

  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // TODO currently this is hardcoded to use GPU 0 but it should try to
  // detect if there is a GPU and which one is best to use.
  const int gpu_idx = 0;
  caffe::Caffe::SetDevice(gpu_idx);
  caffe::Caffe::DeviceQuery();
  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  // open db
  std::string dbname(FLAGS_dbname);
  leveldb::Options options;
  options.error_if_exists = false;
  options.create_if_missing = false;
  options.max_open_files = 100;
  auto db = open_leveldb(dbname, options);
  auto shape = infer_shape(db);
  std::cout << "Inferred shape: " << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;
  // ground truth if available
  auto db_groundtruth = open_leveldb_nofail(FLAGS_dbname_ground_truth, options);

  // datum to be read
  caffe::Datum datum;

  // window to display
  const unsigned int box_height = 60;
  const float scale_sc = shape[2]/2; // factor to multiply steering command with
  IplImage* windowImg = cvCreateImage(cvSize(shape[2], shape[1] + box_height), IPL_DEPTH_8U, shape[0]);

  // Caffe model
  // First load prototxt describing the deployment setup
  caffe::NetParameter network_params;
  caffe::ReadProtoFromTextFile(FLAGS_network_prototxt, &network_params);
  caffe::Net<float> network(network_params);
  // Now load trained weights and copy as applicable
  caffe::NetParameter trained_network_params;
  caffe::ReadNetParamsFromBinaryFileOrDie(FLAGS_network_caffemodel, &trained_network_params);
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
  // Data normalizer
  std::string normalization_fname(FLAGS_normalization_protobinary);
  LinearNormalizer<float> normalizer(normalization_fname);

  // iterate
  auto it = db->NewIterator(leveldb::ReadOptions());
  it->Seek(key_from_int(FLAGS_start_frame));
  unsigned int count = 0;
  unsigned int info_iter = 10;
  unsigned int wait_ms = 100;
  for(; it->Valid(); it->Next())
  {
    datum.ParseFromString(it->value().ToString());

    // clear window
    cvSet(windowImg, cvScalar(0,0,0));
    // draw current frame
    datum_to_ipl(datum, windowImg);

    // predict current frame
    transformer.Transform(datum, input_blob);
    network.Forward();
    normalizer.Denormalize(output_blob);

    // raw output data
    const float* output_data = output_blob->cpu_data();
    // visualize steering command
    float steering_command = output_data[n_outputs - 1];
    cvRectangle(windowImg,
                cvPoint(shape[2] / 2, shape[1]),
                cvPoint(shape[2] / 2 - scale_sc * steering_command, shape[1] + box_height / 2),
                cvScalar(237,99,157), -2);
    // ground truth if available
    if(db_groundtruth != nullptr) {
      std::string gt_value;
      auto status = db_groundtruth->Get(leveldb::ReadOptions(), it->key(), &gt_value);
      CHECK(status.ok()) << status.ToString();
      caffe::Datum gt_datum;
      gt_datum.ParseFromString(gt_value);
      float gt_steering_command = gt_datum.float_data(n_outputs - 1);
      cvRectangle(windowImg,
                  cvPoint(shape[2] / 2, shape[1] + box_height / 2),
                  cvPoint(shape[2] / 2 - scale_sc * gt_steering_command, shape[1] + box_height),
                  cvScalar(99,237,157), -2);
    }

    // use a constant name for the window otherwise a new window is created on each call
    cvShowImage(dbname.c_str(), windowImg);

    count += 1;
    // display and apply user control
    auto key = cvWaitKey(wait_ms);
    auto val = bindings.find(key);
    if(val != bindings.end()) {
      if(val->second == "quit") {
        break;
      } else if(val->second == "fast_forward") {
        wait_ms = std::max((int)wait_ms - 10, 1);
      } else if(val->second == "slow_down") {
        wait_ms = wait_ms + 10;
      } else if(val->second == "jump_hundred_forward") {
        for(int i = 0; i < 100; ++i) {
          if(!it->Valid()) break;
          it->Next();
        }
      } else if(val->second == "jump_hundred_back") {
        for(int i = 0; i < 100; ++i) {
          if(!it->Valid()) break;
          it->Prev();
        }
      }
    }

    if(count % info_iter == 0) std::cout << "Frame: " << it->key().ToString() << std::endl;
  }
  std::cout << "Played a total of " << count << " keys." << std::endl;

  return 0;
}
