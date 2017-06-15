#ifndef panene_linear_scheduler_h
#define panene_linear_scheduler_h

#include <scheduler/scheduler.h>

namespace panene
{

class LinearScheduler : public Scheduler
{
  int w1, w2, w3;

public:
  LinearScheduler() = default;
  LinearScheduler(int w1_, int w2_, int w3_) : w1(w1_), w2(w2_), w3(w3_) { }

  Schedule schedule(int maxOps) {
    if(maxOps == -1) {
      return Schedule(-1, -1, -1);
    }
    else {
      int sum = w1 + w2 + w3;
      return Schedule(maxOps * w1 / sum, maxOps * w2 / sum, maxOps - maxOps * (w1 + w2) / sum);
    }
  }
};

}

#endif
