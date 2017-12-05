#ifndef panene_metadata_h
#define panene_metadata_h

#include <iostream>
#include <fstream>

#include <util/timer.h>
#include <util/matrix.h>
#include <util/result_set.h>
#include <data_source/binary_data_source.h>

#ifdef _WIN32
#define SEP "\\"
#else
#define SEP "/"
#endif

namespace panene
{

struct Dataset {
  std::string base;
  std::string name;
  std::string version;
  std::string path;
  std::string queryPath;
  std::string answerPath;

  std::string getTrainPath() {
    return base + SEP + name + SEP + name + "." + version + ".bin";
  }

  std::string getQueryPath() {
    return base + SEP + name + SEP + "test.bin";
  }

  std::string getAnswerPath() {
    return base + SEP + name + SEP + name + "." + version + ".answer.txt";
  }

  size_t n;
  size_t dim;

  Dataset() = default;
  Dataset(const std::string& base_,
          const std::string& name_, 
          const std::string& version_, 
          size_t n_, size_t dim_) : base(base_), name(name_), version(version_), n(n_), dim(dim_) { 

    path = getTrainPath();
    queryPath = getQueryPath();
    answerPath = getAnswerPath();
  }
};

};

#endif
