#include "utils.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

void check_same_state(const std::vector<leveldb::Iterator*>& its) {
  for(unsigned int i = 1; i < its.size(); ++i) {
    CHECK(its[0]->Valid() == its[i]->Valid()) << "Inconsistent state of Iterators!";
  }
}

// convert integer to key for leveldb
std::string key_from_int(int i) {
  const int width = 8;
  CHECK_LE(i, (int)std::pow(10, width) - 1) <<
    "Index is too large to be compatible with the configured key format. Increase width of key.";
  std::stringstream ss;
  ss << std::setw(width) << std::setfill('0') << i;
  return ss.str();
}


// Divide dataset into trainset consisting of train_size datums and testset
// consisting of the remaining entries
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  if(argc < 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " train_size db1 [db2 ...]";
    return 1;
  }

  unsigned int train_size = atoi(argv[1]);
  int n_dbs = argc - 2;
  std::vector<std::string> dbnames(n_dbs);
  for(int i = 0; i < n_dbs; ++i) {
    dbnames[i] = argv[i + 2];
  }
  unsigned int info_iter = 5000; // frequency of status reports

  // open input dbs
  leveldb::Options options;
  options.error_if_exists = false;
  options.create_if_missing = false;
  options.max_open_files = 100;
  std::vector<leveldb::DB*> dbs(dbnames.size());
  std::vector<leveldb::Iterator*> its(dbnames.size());
  for(int i = 0; i < dbs.size(); ++i) {
    dbs[i] = open_leveldb(dbnames[i], options);
    its[i] = dbs[i]->NewIterator(leveldb::ReadOptions());
    its[i]->SeekToFirst();
  }

  // count and make sure dbs have the same keys
  unsigned int count = 0;
  while(its[0]->Valid()) {
    check_same_state(its);

    // store key
    auto key_0 = its[0]->key().ToString();
    // make sure dbs are synchronized
    for(unsigned int i = 1; i < n_dbs; ++i) {
      auto key_i = its[i]->key().ToString();
      CHECK(key_0 == key_i) << "Differing keys: " << key_0 << " != " << key_i;
    }

    // advance all iterators
    for(unsigned int i = 0; i < n_dbs; ++i) {
      its[i]->Next();
    }
    count += 1;
  }
  check_same_state(its);

  std::cout << "Input dbs size: " << count << std::endl;
  CHECK_LT(train_size, count) << ": Keeping dataset as it is.";

  // now divide
  std::vector<std::string> dbnames_train(n_dbs);
  std::vector<std::string> dbnames_test(n_dbs);
  for(unsigned int i = 0; i < n_dbs; ++i) {
    dbnames_train[i] = dbnames[i] + "_train";
    dbnames_test[i] = dbnames[i] + "_test";
  }
  // prepare output dbs
  leveldb::Options output_options;
  output_options.error_if_exists = true;
  output_options.create_if_missing = true;
  output_options.max_open_files = 100;
  leveldb::WriteOptions write_options;

  // open output dbs
  std::vector<leveldb::DB*> dbs_train(dbnames.size());
  std::vector<leveldb::DB*> dbs_test(dbnames.size());
  for(int i = 0; i < dbs.size(); ++i) {
    dbs_train[i] = open_leveldb(dbnames_train[i], output_options);
    dbs_test[i] = open_leveldb(dbnames_test[i], output_options);
  }

  for(unsigned int i = 0; i < dbs.size(); ++i) {
    its[i]->SeekToFirst();
  }
  // write train set
  for(unsigned int s = 0; s < train_size; ++s) {
    if(s % info_iter == 0) std::cout << "Processed " << s << " entries." << std::endl;
    for(unsigned int i = 0; i < dbs.size(); ++i) {
      auto status = dbs_train[i]->Put(write_options, key_from_int(s), its[i]->value());
      CHECK(status.ok()) << status.ToString();
      its[i]->Next();
    }
  }
  std::cout << "Wrote " << train_size << " datums into train sets ";
  for(unsigned int i = 0; i < dbs.size(); ++i) {
    std::cout << dbnames_train[i];
    if(i + 1 < dbs.size()) std::cout << ", ";
    else std::cout << "." << std::endl;
  }

  // write test set
  unsigned int s = 0;
  for(; its[0]->Valid(); ++s) {
    if(s % info_iter == 0) std::cout << "Processed " << s << " entries." << std::endl;
    for(unsigned int i = 0; i < dbs.size(); ++i) {
      auto status = dbs_test[i]->Put(write_options, key_from_int(s), its[i]->value());
      CHECK(status.ok()) << status.ToString();
      its[i]->Next();
    }
  }
  std::cout << "Wrote " << s << " datums into test sets ";
  for(unsigned int i = 0; i < dbs.size(); ++i) {
    std::cout << dbnames_test[i];
    if(i + 1 < dbs.size()) std::cout << ", ";
    else std::cout << "." << std::endl;
  }

  return 0;
}
