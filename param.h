#ifndef PARAM_H
#define PARAM_H

#include <flann/flann.hpp>

using namespace std;
using namespace flann;

class Param
{
public:
    Param(IndexParams indexParam_) : indexParam(indexParam_) {};
    string format();
    string algorithm();
    IndexParams& getIndexParam();

protected:
    IndexParams indexParam;
};

#endif