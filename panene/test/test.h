#ifndef panene_test_h
#define panene_test_h

#include <string>

namespace panene {

class Test
{
public:
  virtual std::string getName() = 0;
  virtual void run() = 0;
};

}

#endif

