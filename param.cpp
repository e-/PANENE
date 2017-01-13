#include "param.h"
#include <sstream>
#include <string>

string Param::format() {
    IndexParams& ip = this -> indexParam;
    flann_algorithm_t alg = get_param<flann_algorithm_t>(ip, "algorithm");
    stringstream ss;
    string result;
    if(alg == FLANN_INDEX_KDTREE) {
        ss << "KDTree(trees=" << get_param<int>(ip, "trees") << ")";
    }
    else if(alg == FLANN_INDEX_KMEANS) {
        ss << "Kmeans(branch=" << get_param<int>(ip, "branching") << ")";
    }
    else {
        ss << "Unspecified Parameters";
    }

    getline(ss, result);
    return result;
}

string Param::algorithm() {
    IndexParams& ip = this -> indexParam;
    flann_algorithm_t alg = get_param<flann_algorithm_t>(ip, "algorithm");
    stringstream ss;
    string result;
    if(alg == FLANN_INDEX_KDTREE) {
        ss << "KDTree";
    }
    else if(alg == FLANN_INDEX_KMEANS) {
        ss << "Kmeans";
    }
    else {
        ss << "Unspecified Parameters";
    }

    getline(ss, result);
    return result;
}

IndexParams& Param::getIndexParam() {return this -> indexParam;}