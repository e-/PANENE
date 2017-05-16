#ifndef panene_naive_scheduler_h
#define panene_naive_scheduler_h

#include "schedule.h"

namespace panene
{

class NaiveScheduler
{
public:
  NaiveScheduler() = default;

  Schedule schedule(int maxOps) {
    if(maxOps == -1) {
      return Schedule(-1, -1, -1);
    }
    else {
      return Schedule(maxOps / 3, maxOps / 3, maxOps - maxOps / 3 * 2);
    }
  }
};

}

#endif
