#include "utils.h"

#include <caffe/data_transformer.hpp>

#include <iostream>

#include <sys/shm.h>

const int ASCII_ESC = 27;
const int n_outputs = 15;
const float scale_sc = 300; // factor to multiply steering command with

const int n_channels = 3;
// image size as captured from torcs
const int image_width = 640;
const int image_height = 480;
// image size used for network input
const int net_image_width = 280;
const int net_image_height = 210;

// struct used for shared memory communication with torcs
struct SharedStruct
{  
    int written;  //a label, if 1: available to read, if 0: available to write
    uint8_t data[image_width*image_height*3];  // image data field  
    int control;
    int pause;
    double fast;

    double dist_L;
    double dist_R;

    double toMarking_L;
    double toMarking_M;
    double toMarking_R;

    double dist_LL;
    double dist_MM;
    double dist_RR;

    double toMarking_LL;
    double toMarking_ML;
    double toMarking_MR;
    double toMarking_RR;

    double toMiddle;
    double angle;
    double speed;

    double steerCmd;
    double accelCmd;
    double brakeCmd;

    void clear() {
      written = 0;
      control = 0;
      pause = 0;
      fast = 0;
      dist_L = 0;
      dist_R = 0;
      toMarking_L = 0;
      toMarking_M = 0;
      toMarking_R = 0;
      dist_LL = 0;
      dist_MM = 0;
      dist_RR = 0;
      toMarking_LL = 0;
      toMarking_ML = 0;
      toMarking_MR = 0;
      toMarking_RR = 0;
      toMiddle = 0;
      angle = 0;
      speed = 0;
      
      steerCmd = 0;
      accelCmd = 0;
      brakeCmd = 0;
    }

    void init_datum(caffe::Datum& datum) {
      datum.set_channels(n_channels);
      datum.set_height(net_image_height);
      datum.set_width(net_image_width);
      datum.mutable_data()->resize(3 * net_image_height * net_image_width); 
    }
    void copy_img_to_datum(caffe::Datum& datum) {
      CHECK_EQ(datum.channels(), n_channels);
      CHECK_EQ(datum.height(), net_image_height);
      CHECK_EQ(datum.width(), net_image_width);
      // for testing we use the same method as used in torcs - then we will
      // switch to c++ binding for opencv and compare the result
      IplImage* input_img = cvCreateImage(cvSize(image_width, image_height), IPL_DEPTH_8U, n_channels);
      IplImage* net_img = cvCreateImage(cvSize(net_image_width, net_image_height), IPL_DEPTH_8U, n_channels);
      // copy input image to IplImage
      //for(int i = 0; i < image_height * image_width * n_channels; ++i) {
      //  input_img->imageData[i] = data[i];
      //}
      for(int h = 0; h < image_height; ++h) {
        for(int w = 0; w < image_width; ++w) {
          for(int c = 0; c < n_channels; ++c) {
            input_img->imageData[(h*image_width + w)*n_channels + c] = data[((image_height - 1 - h)*image_width + w)*n_channels + (n_channels - 1 - c)];
          }
        }
      }
      // resize
      cvResize(input_img, net_img);
      cvShowImage("Frame", net_img);

      // copy resized image into datum
      //datum.mutable_data()->resize(3 * net_image_height * net_image_width); 
      std::string* datum_data = datum.mutable_data();
      for(int h = 0; h < net_image_height; ++h) {
        for(int w = 0; w < net_image_width; ++w) {
          for(int c = 0; c < n_channels; ++c) {
            (*datum_data)[(c*net_image_height + h)*net_image_width + w] = (char)(net_img->imageData[(h*net_image_width + w)*n_channels + c]);
          }
        }
      }

      // clean up
      cvReleaseImage(&input_img);
      cvReleaseImage(&net_img);
    }

    void apply_speed_control(const float desired_speed) {
      if (desired_speed >= speed) {
        accelCmd = 0.2*(desired_speed - speed + 1);
        if(accelCmd > 1) accelCmd = 1.0;
        brakeCmd = 0.0;
      } else {
        brakeCmd = 0.1*(speed - desired_speed);
        if(brakeCmd > 1) brakeCmd = 1.0;
        accelCmd = 0.0;
      }
    }
};



// Show frames in leveldb
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  if(argc != 4) {
    LOG(ERROR) << "Usage: " << argv[0] << " input_network input_weights normalization_param_blob";
    return 1;
  }

  // TODO currently this is hardcoded to use GPU 0 but it should try to
  // detect if there is a GPU and which one is best to use.
  const int gpu_idx = 0;
  caffe::Caffe::SetDevice(gpu_idx);
  caffe::Caffe::DeviceQuery();
  caffe::Caffe::set_mode(caffe::Caffe::GPU);
  // setup datum to be read
  caffe::Datum datum;

  // Caffe model
  // First load prototxt describing the deployment setup
  caffe::NetParameter network_params;
  caffe::ReadProtoFromTextFile(argv[1], &network_params);
  caffe::Net<float> network(network_params);
  // Now load trained weights and copy as applicable
  caffe::NetParameter trained_network_params;
  caffe::ReadNetParamsFromBinaryFileOrDie(argv[2], &trained_network_params);
  network.CopyTrainedLayersFrom(trained_network_params);

  // expected shape
  std::vector<int> shape{n_channels, net_image_height, net_image_width};

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
  std::string normalization_fname(argv[3]);
  LinearNormalizer<float> normalizer(normalization_fname);

  // Shared memory - see also man shmget
  key_t shm_key = (key_t)4567;  // torcs must use the same
  int perm = 0600;             // permission mode
  int shm_flags = IPC_CREAT | perm;
  int shm_id = shmget(shm_key, sizeof(SharedStruct), shm_flags);
  CHECK(shm_id != -1) << "shmget() unsuccessful.";
  SharedStruct* shm_struct = (SharedStruct*)shmat(shm_id, 0, 0);
  CHECK(shm_struct != (SharedStruct*)-1) << "shmat() unsuccessful.";
  shm_struct->clear();
  shm_struct->init_datum(datum);

  // Visalization of steering angle
  const unsigned int box_width = 280;
  const unsigned int box_height = 60;
  const float scale_sc = 300; // factor to multiply steering command with
  IplImage* window_img = cvCreateImage(cvSize(box_width, box_height), IPL_DEPTH_8U, 3);
  cvSet(window_img, cvScalar(0,0,0));
  cvShowImage("Steering Command", window_img);

  float desired_speed = 10;
  while(1) {
    if(shm_struct->written == 1) {
      // load image into datum to be able to apply transformer
      shm_struct->copy_img_to_datum(datum);
      // predict current frame
      transformer.Transform(datum, input_blob);
      network.Forward();
      normalizer.Denormalize(output_blob);
      // raw output data
      const float* output_data = output_blob->cpu_data();
      // visualize steering command
      float steering_command = output_data[n_outputs - 1];

      // apply steering command
      shm_struct->steerCmd = steering_command;
      // apply speed control
      shm_struct->apply_speed_control(desired_speed);

      // visualize
      cvSet(window_img, cvScalar(0,0,0));
      cvRectangle(window_img,
                  cvPoint(box_width / 2, 0),
                  cvPoint(box_width / 2 - scale_sc * steering_command, box_height),
                  cvScalar(237,99,157), -2);

      // use a constant name for the window otherwise a new window is created on each call
      cvShowImage("Steering Command", window_img);

      // signal that data is stale
      shm_struct->written = 0;
    }

    auto key = cvWaitKey(1);
    if(key == ASCII_ESC || key == 'q') {
      shm_struct->pause = 0;
      break;
    } else if(key == 'p') {
      shm_struct->pause = 1 - shm_struct->pause;
    } else if(key == 'w') {
      shm_struct->accelCmd = 0.2;
      shm_struct->brakeCmd = 0;
    } else if(key == 's') {
      shm_struct->accelCmd = 0;
      shm_struct->brakeCmd = 0.5;
    } else if(key == 'a') {
      shm_struct->steerCmd = +0.25;
    } else if(key == 'd') {
      shm_struct->steerCmd = -0.25;
    } else if(key == '+') {
      desired_speed += 1;
    } else if(key == '-') {
      desired_speed -= 1;
    }


  }

  return 0;
}
