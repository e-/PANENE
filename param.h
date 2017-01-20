#ifndef PARAM_H
#define PARAM_H

#include <flann/flann.hpp>
#include <string>

using namespace std;
using namespace flann;

struct Dataset {
    Dataset(string path_, string queryPath_, int initialSize_, int chunkSize_, int chunkN_, int dim_): path(path_), initialSize(initialSize_), chunkSize(chunkSize_), chunkN(chunkN_), dim(dim_), queryPath(queryPath_) {
        
    }

    string path;
    string queryPath;
    int initialSize;
    int chunkSize;
    int chunkN;
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
