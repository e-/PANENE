#include "timer.h"

void Timer::begin() {clock_gettime(CLOCK_MONOTONIC, &bb);}
double Timer::end() {
    clock_gettime(CLOCK_MONOTONIC, &ff);
    double elapsed = ff.tv_sec - bb.tv_sec;
    elapsed += (ff.tv_nsec - bb.tv_nsec) / 1000000000.0;
    return elapsed;
}