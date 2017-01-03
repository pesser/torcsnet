#include "utils.h"

#include <iostream>

const int ASCII_ESC = 27;

// split a leveldb that contains caffe::Datums with float_data into two
// leveldb databases, the first containing the image in a Datum, the other
// containing the float_data as a Datum. Then it is easier to perform
// regression because using a data layer, input datums that have no
// uint8 data will be processed as float_data.
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  if(argc != 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " input_db out_prefix";
    return 1;
  }

  // open input db
  std::string dbname(argv[1]);
  leveldb::Options options;
  options.error_if_exists = false;
  options.create_if_missing = false;
  options.max_open_files = 100;
  auto db = open_leveldb(dbname, options);
  auto shape = infer_shape(db);
  auto float_data_size = infer_float_data_size(db);
  std::cout << "Inferred shape: " << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;
  std::cout << "Inferred float_data_size: " << float_data_size << std::endl;
  CHECK(float_data_size > 0) << "Can not split dataset which contains no float data.";

  // create output dbs
  leveldb::Options output_options;
  output_options.error_if_exists = true;
  output_options.create_if_missing = true;
  output_options.max_open_files = 100;

  // db containing the inputs
  std::string input_dbname = std::string(argv[2]) + "_input";
  auto input_db = open_leveldb(input_dbname, output_options);

  // db containing the targets
  std::string target_dbname = std::string(argv[2]) + "_target";
  auto target_db = open_leveldb(target_dbname, output_options);

  // datum containing input together with targets in float_data field
  caffe::Datum original_datum;
  // the input datum
  caffe::Datum input_datum;
  input_datum.set_channels(shape[0]);
  input_datum.set_height(shape[1]);
  input_datum.set_width(shape[2]);
  // the target datum
  caffe::Datum target_datum;
  target_datum.set_channels(1);
  target_datum.set_height(1);
  target_datum.set_width(float_data_size);

  leveldb::WriteOptions write_options;
  std::string serialized_datum;

  // iterate over original db
  auto it = db->NewIterator(leveldb::ReadOptions());
  unsigned int count = 0;
  unsigned int info_iter = 5000;
  for(it->SeekToFirst(); it->Valid(); it->Next())
  {
    if(count % info_iter == 0) {
      std::cout << "Processed " << count << " entries." << std::endl;
    }
    count += 1;
    std::string key = it->key().ToString();

    original_datum.ParseFromString(it->value().ToString());

    // copy data to input datum
    input_datum.set_data(original_datum.data());
    input_datum.SerializeToString(&serialized_datum);
    input_db->Put(write_options, key, serialized_datum);

    // copy float data to target datum
    *(target_datum.mutable_float_data()) = original_datum.float_data();
    target_datum.SerializeToString(&serialized_datum);
    target_db->Put(write_options, key, serialized_datum);
  }
  std::cout << "Split a total of " << count << " keys." << std::endl;

  return 0;
}
