#ifndef logger_h
#define logger_h

#include <iostream>

namespace panene {

class Logger() {
  static void Log(string s) {
    cerr << s;
  }
}
#endif
