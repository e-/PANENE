#ifndef TIMER_H
#define TIMER_H

#include <ctime>

namespace panene {

class Timer
{
public:
  Timer() {}
  void begin() {
    clock_gettime(CLOCK_MONOTONIC, &bb);
  }

  double end() {
    clock_gettime(CLOCK_MONOTONIC, &ff);
    double elapsed = ff.tv_sec - bb.tv_sec;
    elapsed += (ff.tv_nsec - bb.tv_nsec) / 1000000000.0;
    return elapsed;
  }

protected:
  struct timespec bb, ff;
};

}

#endif
