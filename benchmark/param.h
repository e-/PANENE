#ifndef PARAM_H
#define PARAM_H

#include <flann/flann.hpp>
#include <string>

using namespace std;
using namespace flann;

struct Dataset {
    Dataset(string name_, string path_, string queryPath_, int initialSize_, int chunkSize_, int chunkN_, int querySize_, int dim_): 
      name(name_),
      path(path_), queryPath(queryPath_), initialSize(initialSize_), chunkSize(chunkSize_), chunkN(chunkN_), querySize(querySize_), dim(dim_) {
        
    }

    string name;
    string path;
    string queryPath;
    int initialSize;
    int chunkSize;
    int chunkN;
    int querySize;
    int dim;
};

class FLANNParam
{
public:
    FLANNParam(IndexParams indexParams_, SearchParams searchParams_, Dataset dataset_) : indexParams(indexParams_), searchParams(searchParams_), dataset(dataset_) {};
    string format();
    string algorithm();
    IndexParams& getIndexParams();
    SearchParams& getSearchParams();
    Dataset& getDataset();

protected:
    IndexParams indexParams;
    SearchParams searchParams;
    Dataset dataset;
};

#endif
