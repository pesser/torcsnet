#include "utils.h"

#include <iostream>

// Normalize float data to a specified interval and print parameters for
// (de-)normalization. If out_db is specified, write normalized data into
// it.
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  if(argc != 4 && argc != 5) {
    LOG(ERROR) << "Usage: " << argv[0] << " a b input_db [out_db]";
    return 1;
  }

  // interval to normalize to
  float a = atof(argv[1]),
        b = atof(argv[2]);
  CHECK(a < b) << "Invalid interval: [" << a << ", " << b << "]";

  // open input db
  std::string dbname(argv[3]);
  leveldb::Options options;
  options.error_if_exists = false;
  options.create_if_missing = false;
  options.max_open_files = 100;
  auto db = open_leveldb(dbname, options);
  auto shape = infer_shape(db);
  auto float_data_size = infer_float_data_size(db);
  std::cout << "Inferred shape: " << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;
  std::cout << "Inferred float_data_size: " << float_data_size << std::endl;
  CHECK(float_data_size > 0) << "Can not normalize dataset which contains no float data.";

  // collect min and max of float fields
  std::vector<float> mins(float_data_size),
                     maxs(float_data_size);

  // initialize with first datum
  caffe::Datum datum;
  auto it = db->NewIterator(leveldb::ReadOptions());
  it->SeekToFirst();
  datum.ParseFromString(it->value().ToString());
  for(unsigned int i = 0; i < float_data_size; ++i) {
    mins[i] = datum.float_data(i);
  }

  // iterate over rest
  it->Next();
  unsigned int count = 1;
  unsigned int info_iter = 5000;
  for(; it->Valid(); it->Next())
  {
    datum.ParseFromString(it->value().ToString());
    for(unsigned int i = 0; i < float_data_size; ++i) {
      mins[i] = std::min<float>(mins[i], datum.float_data(i));
      maxs[i] = std::max<float>(maxs[i], datum.float_data(i));
    }
    if(count % info_iter == 0) {
      std::cout << "Processed " << count << " entries." << std::endl;
    }
    count += 1;
  }
  std::cout << "Looked at a total of " << count << " keys." << std::endl;

  // compute slope and bias for each field
  std::vector<float> normalization_parameters(2 * float_data_size);
  for(unsigned int  i = 0; i < float_data_size; ++i) {
    normalization_parameters[i*2 + 0] = (b - a)/(maxs[i] - mins[i]);
    normalization_parameters[i*2 + 1] = a - (b - a)/(maxs[i] - mins[i])*mins[i];
  }

  // print statistics and (de-)normalization functions
  std::cout << std::fixed;
  std::cout.precision(6);
  std::cout << "// Statistics of float_data fields:" << std::endl;
  std::cout << std::endl << "// Mins:" << std::endl << "// ";
  for (int i = 0; i < float_data_size; ++i) {
    std::cout << std::setw(12) << mins[i];
  }
  std::cout << std::endl << "// Maxs:" << std::endl << "//";
  for (int i = 0; i < float_data_size; ++i) {
    std::cout << std::setw(12) << maxs[i];
  }
  std::cout << std::endl << "// " << std::endl << "// ";
  std::cout << "// Rows describing slope and bias to normalize into [" << a << ", " << b << "]." << std::endl;
  std::cout << "const unsigned int float_data_size = " << float_data_size << ";" << std::endl;
  std::cout << "const float normalization_parameters[] = {" << std::endl;
  for(unsigned int  i = 0; i < float_data_size; ++i) {
    std::cout << "\t" << normalization_parameters[i*2 + 0] << ", " << normalization_parameters[i*2 + 1];
    if(i + 1 < float_data_size) std::cout << ",";
    else std::cout << "};";
    std::cout << std::endl;
  }
  std::cout <<
    "template <class T>" << std::endl <<
    "float normalize(unsigned int i, T x) {" << std::endl <<
    "  assert(i < float_data_size);" << std::endl <<
    "  return normalization_parameters[i * 2 + 0] * x + normalization_parameters[i * 2 + 1];" << std::endl <<
    "}" << std::endl;
  std::cout <<
    "template <class T>" << std::endl <<
    "float denormalize(unsigned int i, T y) {" << std::endl <<
    "  assert(i < float_data_size);" << std::endl <<
    "  return 1.0 / normalization_parameters[i * 2 + 0] * (y - normalization_parameters[i * 2 + 1]);" << std::endl <<
    "}" << std::endl;

  if(argc == 4) {
    // done
    return 0;
  }

  // write normalized data to db
  leveldb::Options output_options;
  output_options.error_if_exists = true;
  output_options.create_if_missing = true;
  output_options.max_open_files = 100;
  std::string out_dbname = std::string(argv[4]);
  auto out_db = open_leveldb(out_dbname, output_options);
  std::string serialized_datum;
  leveldb::WriteOptions write_options;

  // iterate again over input database, normalize it and write to db
  it->SeekToFirst();
  count = 0;
  for(; it->Valid(); it->Next()) {
    datum.ParseFromString(it->value().ToString());
    for(unsigned int i = 0; i < float_data_size; ++i) {
      datum.set_float_data(i, normalization_parameters[i * 2 + 0] * datum.float_data(i) + normalization_parameters[i * 2 + 1]);
    }
    datum.SerializeToString(&serialized_datum);
    out_db->Put(write_options, it->key().ToString(), serialized_datum);

    if(count % info_iter == 0) {
      std::cout << "Processed " << count << " entries." << std::endl;
    }
    count += 1;
  }
  std::cout << "Wrote a total of " << count << " normalized entries to " << out_dbname << "." << std::endl;

  return 0;
}
