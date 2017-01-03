#include "utils.h"

#include <iostream>
#include <random>

void check_same_state(const std::vector<leveldb::Iterator*>& its) {
  for(unsigned int i = 1; i < its.size(); ++i) {
    CHECK(its[0]->Valid() == its[i]->Valid()) << "Inconsistent state of Iterators!";
  }
}


// Shuffle databases in lockstep
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  if(argc < 3) {
    LOG(ERROR) << "Usage: " << argv[0] << " db1 [db2 ...]";
    return 1;
  }

  std::mt19937 random_engine{std::random_device{}()};

  int n_dbs = argc - 1;
  // open input dbs
  std::vector<std::string> dbnames(n_dbs);
  for(int i = 0; i < n_dbs; ++i) {
    dbnames[i] = argv[i + 1];
  }
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

  // collect all keys
  std::vector<std::string> keys;
  while(its[0]->Valid()) {
    check_same_state(its);

    // store key
    auto key_0 = its[0]->key().ToString();
    keys.push_back(key_0);

    // make sure dbs are synchronized
    for(unsigned int i = 1; i < n_dbs; ++i) {
      auto key_i = its[i]->key().ToString();
      CHECK(key_0 == key_i) << "Differing keys: " << key_0 << " != " << key_i;
    }

    // advance all iterators
    for(unsigned int i = 0; i < n_dbs; ++i) {
      its[i]->Next();
    }
  }
  check_same_state(its);

  // shuffle
  std::vector<std::string> shuffled_keys(keys);
  std::shuffle(shuffled_keys.begin(), shuffled_keys.end(), random_engine);

  // prepare output dbs
  leveldb::Options output_options;
  output_options.error_if_exists = true;
  output_options.create_if_missing = true;
  output_options.max_open_files = 100;
  leveldb::WriteOptions write_options;
  leveldb::ReadOptions read_options;
  std::vector<std::string> out_dbnames(n_dbs);
  std::vector<leveldb::DB*> out_dbs(dbnames.size());
  for(int i = 0; i < n_dbs; ++i) {
    out_dbnames[i] = dbnames[i] + "_shuffled";
    out_dbs[i] = open_leveldb(out_dbnames[i], output_options);
  }

  // write original keys with shuffled data
  std::string value;
  unsigned int count = 0;
  unsigned int info_iter = 5000;
  for(unsigned int i = 0; i < keys.size(); ++i) {
    for(unsigned int db = 0; db < n_dbs; ++db) {
      auto s = dbs[db]->Get(read_options, shuffled_keys[i], &value);
      CHECK(s.ok()) << s.ToString();
      s = out_dbs[db]->Put(write_options, keys[i], value);
      CHECK(s.ok()) << s.ToString();
    }
    if(count % info_iter == 0) {
      std::cout << "Shuffled " << count << " entries." << std::endl;
    }
    count += 1;
  }
  std::cout << "Shuffled a total of " << keys.size() << " entries into ";
  for(unsigned int i = 0; i < n_dbs; ++i) {
    std::cout << out_dbnames[i];
    if(i + 1 < n_dbs) std::cout << ", ";
    else std::cout << "." << std::endl;
  }

  return 0;
}
