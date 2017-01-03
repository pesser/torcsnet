#include "utils.h"

#include <iostream>

const int ASCII_ESC = 27;

// Show frames in leveldb
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  if(argc != 2) {
    LOG(ERROR) << "Usage: " << argv[0] << " input_db";
    return 1;
  }

  std::string dbname(argv[1]);
  leveldb::Options options;
  options.error_if_exists = false;
  options.create_if_missing = false;
  options.max_open_files = 100;
  auto db = open_leveldb(dbname, options);
  auto shape = infer_shape(db);
  std::cout << "Inferred shape: " << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;

  caffe::Datum datum;
  IplImage* windowImg = cvCreateImage(cvSize(shape[2], shape[1]), IPL_DEPTH_8U, shape[0]);

  auto it = db->NewIterator(leveldb::ReadOptions());
  unsigned int count = 0;
  for(it->SeekToFirst(); it->Valid(); it->Next())
  {
    count += 1;
    datum.ParseFromString(it->value().ToString());
    datum_to_ipl(datum, windowImg);
    // use a constant name for the window otherwise a new window is created on each call
    cvShowImage(dbname.c_str(), windowImg);
    
    auto key = cvWaitKey(20);
    if(key == ASCII_ESC) break;
  }
  std::cout << "Played a total of " << count << " keys." << std::endl;

  return 0;
}
