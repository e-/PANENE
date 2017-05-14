#ifndef panene_test_h
#define panene_test_h

namespace panene {
using namespace std;

class KDTreeTest : Test
{
public:
  virtual string getName() = 0;
  virtual void run() = 0;
}

}

#endif

