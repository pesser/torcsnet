#pragma once

#include <leveldb/db.h>

#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <caffe/caffe.hpp>

auto open_leveldb(const std::string& dbname, const leveldb::Options& options)
{
  leveldb::DB* db;
  LOG(INFO) << "Opening " << dbname;
  auto status = leveldb::DB::Open(options, dbname, &db);
  CHECK(status.ok()) << "Failed to open leveldb: " << dbname;
  return db;
}

// return (channels, height, width) of first image datum in leveldb
std::vector<unsigned int> infer_shape(leveldb::DB* db)
{
  caffe::Datum datum;
  auto it = db->NewIterator(leveldb::ReadOptions());
  it->SeekToFirst();
  datum.ParseFromString(it->value().ToString());
  return {(unsigned int)datum.channels(), (unsigned int)datum.height(), (unsigned int)datum.width()};
}


// return float_data_size of first datum in leveldb
unsigned int infer_float_data_size(leveldb::DB* db)
{
  caffe::Datum datum;
  auto it = db->NewIterator(leveldb::ReadOptions());
  it->SeekToFirst();
  datum.ParseFromString(it->value().ToString());
  return (unsigned int)datum.float_data_size();
}


// copy datum image (channel, height, width) into IplImage (height, width,
// channel)
void datum_to_ipl(const caffe::Datum& datum, IplImage* img)
{
  unsigned int n_channels = datum.channels(),
               height = datum.height(),
               width = datum.width();
  const std::string& data = datum.data();

  CHECK(n_channels == img->nChannels && width == img->width && height <= img->height) << "Incompatible dimensions. "
    << "(" << n_channels << ", " << img->nChannels << "), (" << width << ", " << img->width << "), "
    << "(" << height << ", " << img->height << ")";

  for(unsigned int h = 0; h < height; ++h) {
    for(unsigned int w = 0; w < width; ++w) {
      for(unsigned int channel = 0; channel < n_channels; ++channel) {
        img->imageData[(h * width + w) * n_channels + channel] = (uint8_t)data[(channel * height + h) * width + w];
      }
    }
  }
}
