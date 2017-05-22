#ifndef panene_data_source_h
#define panene_data_source_h

#include <string>

namespace panene
{

class DataSource2 {
public:
  DataSource2() = default;

  DataSource2(std::string path_, std::string name_, int n_, int d_) : path(path_), name(name_), n(n_), d(d_) {

  }

  std::string path;
  std::string name;
  int n;
  int d;
};

}

#endif

