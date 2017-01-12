#ifndef TIMER_H
#define TIMER_H

#include <ctime>

using namespace std;

class Timer
{
public:
  Timer() {}
  void begin();
  double end();

protected:
  struct timespec bb, ff;
};

#endif