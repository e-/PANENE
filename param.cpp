#include "param.h"
#include <sstream>
#include <string>

string splitCriteria(int a) {
    if (a == FLANN_MEAN) return "MEAN";
    return "MEDIAN";
}

string updateCriteria(int a) {
    if(a == FLANN_HEIGHT_DIFFERENCE) return "Height";
    return "AVG_Depth";
}

string FLANNParam::format() {
    IndexParams& ip = this -> indexParams;
    flann_algorithm_t alg = get_param<flann_algorithm_t>(ip, "algorithm");
    stringstream ss;
    string result;
    if(alg == FLANN_INDEX_KDTREE) {
        ss << "KDTree(trees=" << get_param<int>(ip, "trees") << ")";
    }
    else if(alg == FLANN_INDEX_KDTREE_BALANCED) {
        ss << "KDBalancedTree(trees=" << get_param<int>(ip, "trees") << ',' 
          << get_param<float>(ip, "rebalance_threshold") << ',' 
          << updateCriteria(get_param<int>(ip, "update_criteria")) << ',' 
          << splitCriteria(get_param<int>(ip, "split_criteria")) << ")";
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

string FLANNParam::algorithm() {
    IndexParams& ip = this -> indexParams;
    flann_algorithm_t alg = get_param<flann_algorithm_t>(ip, "algorithm");
    stringstream ss;
    string result;
    if(alg == FLANN_INDEX_KDTREE) {
        ss << "KDTree";
    }
    else if(alg == FLANN_INDEX_KDTREE_BALANCED) {
        ss << "KDBalancedTree";
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

IndexParams& FLANNParam::getIndexParams() {return this -> indexParams;}

SearchParams& FLANNParam::getSearchParams() {return this -> searchParams;}

Dataset& FLANNParam::getDataset() {return this -> dataset;}
