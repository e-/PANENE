#ifndef logger_h
#define logger_h

#include <iostream>

namespace paknn {

class Logger() {
  static void Log(string s) {
    cerr << s;
  }
}
#endif
