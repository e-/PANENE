#ifndef panene_scheduler_h
#define panene_scheduler_h

#include <scheduler/schedule.h>

namespace panene
{

class Scheduler
{
public:
  Scheduler() = default;

  virtual Schedule schedule(int maxOps) {
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
