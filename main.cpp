#include <flann/flann.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <memory>
#include <set>

#include "timer.h"
#include "param.h"

#include "annoylib.h"

using namespace std;
using namespace flann;

void readData(const string &path, int rows, int cols, float *dataset) {
    ifstream ifs(path);

    for(int i = 0; i < rows * cols; ++i) {
        ifs >> dataset[i]; 
    }
}

float * shuffle(float *dataset, int rows, int cols) {
    UniqueRandom ur(rows);
    float *shuffled = new float[rows * cols];

    for(int i = 0; i < rows; ++i) {
        int d = ur.next();
        for(int j = 0; j < cols; ++j) {
            shuffled[i * cols + j] = dataset[d * cols + j];
        }
    }
    return shuffled;
}

void getCorrectAnswers(Matrix<float> &dataset, int rows, Matrix<float> &queryset, Matrix<int> &answers){
    L2<float> distance = L2<float>();
    typedef typename L2<float>::ResultType DistanceType;

    int n = answers.cols;

#pragma omp parallel for
    for(unsigned int q = 0; q < queryset.rows; q++) {
        int* match = new int[n];
        DistanceType* dists = new DistanceType[n];

        float * query = queryset[q];

        dists[0] = distance(dataset[0], query, dataset.cols);
        match[0] = 0;
        int dcnt = 1;

        for (int i=1; i<rows; ++i) {
            DistanceType tmp = distance(dataset[i], query, dataset.cols);

            if (dcnt<n) {
                match[dcnt] = i;
                dists[dcnt++] = tmp;
            }
            else if (tmp < dists[dcnt-1]) {
                dists[dcnt-1] = tmp;
                match[dcnt-1] = i;
            }

            int j = dcnt-1;
            // bubble up
            while (j>=1 && dists[j]<dists[j-1]) {
                std::swap(dists[j],dists[j-1]);
                std::swap(match[j],match[j-1]);
                j--;
            }
        }

        for (int i=0; i<n; ++i) {
            answers[q][i] = match[i];
        }

        delete[] match;
        delete[] dists;
    }
}

#define GLOVE_INITIAL_SIZE 1000
#define GLOVE_CHUNK_SIZE 1000
#define GLOVE_CHUNK_N 100
#define GLOVE_QUERY_SIZE 1000

#define GLOVE_ORI Dataset("glove.original", "data/glove.trim.txt", "data/glove.query.txt", GLOVE_INITIAL_SIZE, GLOVE_CHUNK_SIZE, GLOVE_CHUNK_N, GLOVE_QUERY_SIZE, 100)
#define GLOVE_SORTED Dataset("glove.sorted", "data/glove.sorted.txt", "data/glove.query.txt", GLOVE_INITIAL_SIZE, GLOVE_CHUNK_SIZE, GLOVE_CHUNK_N, GLOVE_QUERY_SIZE, 100)
#define GLOVE_HALF_SORTED Dataset("glove.halfsorted", "data/glove.halfsorted.txt", "data/glove.query.txt", GLOVE_INITIAL_SIZE, GLOVE_CHUNK_SIZE, GLOVE_CHUNK_N, GLOVE_QUERY_SIZE, 100)


#define SIFT_INITIAL_SIZE 1000
#define SIFT_CHUNK_SIZE 1000
#define SIFT_CHUNK_N 100
#define SIFT_QUERY_SIZE 1000

#define SIFT_ORI Dataset("sift.original", "data/sift.trim.txt", "data/sift.query.txt", SIFT_INITIAL_SIZE, SIFT_CHUNK_SIZE, SIFT_CHUNK_N, SIFT_QUERY_SIZE, 128)
#define SIFT_SORTED Dataset("sift.sorted", "data/sift.sorted.txt", "data/sift.query.txt", SIFT_INITIAL_SIZE, SIFT_CHUNK_SIZE, SIFT_CHUNK_N, SIFT_QUERY_SIZE, 128)
#define SIFT_HALF_SORTED Dataset("sift.halfsorted", "data/sift.halfsorted.txt", "data/sift.query.txt", SIFT_INITIAL_SIZE, SIFT_CHUNK_SIZE, SIFT_CHUNK_N, SIFT_QUERY_SIZE, 128)

int main(int argc, char** argv)
{
    srand(time(NULL));
    int nn = 20;
    int flannRepeat = 3;
    int annoyRepeat = 10;

    Timer timer;
    float *rawDataset, *queryDataset;
    
    int annoySearchParam = 128;
    vector<int> annoyTrees{10, 20, 40, 80, 160};
    int trees = 8;

    vector<FLANNParam> params;
    SearchParams searchParam(512);
    searchParam.cores = 1;

    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 10000), searchParam, GLOVE_ORI));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.1f), searchParam, GLOVE_ORI));

//    params.push_back(FLANNParam(KDTreeIndexParams(trees), searchParam, GLOVE_ORI));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.00001f), searchParam, GLOVE_ORI));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.1f), searchParam, GLOVE_ORI));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 10000), searchParam, GLOVE_ORI));

//    params.push_back(FLANNParam(KDTreeIndexParams(trees), searchParam, GLOVE_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.00001f), searchParam, GLOVE_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.1f), searchParam, GLOVE_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 10000), searchParam, GLOVE_SORTED));

//    params.push_back(FLANNParam(KDTreeIndexParams(trees), searchParam, GLOVE_HALF_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.00001f), searchParam, GLOVE_HALF_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.1f), searchParam, GLOVE_HALF_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 10000), searchParam, GLOVE_HALF_SORTED));

//    params.push_back(FLANNParam(KDTreeIndexParams(trees), searchParam, SIFT_ORI));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.00001f), searchParam, SIFT_ORI));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.1f), searchParam, SIFT_ORI));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 10000), searchParam, SIFT_ORI));

//    params.push_back(FLANNParam(KDTreeIndexParams(trees), searchParam, SIFT_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.00001f), searchParam, SIFT_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.1f), searchParam, SIFT_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 10000), searchParam, SIFT_SORTED));

//    params.push_back(FLANNParam(KDTreeIndexParams(trees), searchParam, SIFT_HALF_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.00001f), searchParam, SIFT_HALF_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.1f), searchParam, SIFT_HALF_SORTED));
    params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 10000), searchParam, SIFT_HALF_SORTED));
 
 
    
    //params.push_back(FLANNParam(KDTreeBalancedIndexParams(trees, 1.1f, FLANN_MEDIAN), searchParam, GLOVE_ORI));

//    params.push_back(Param(KDTreeIndexParams(1)));
//    params.push_back(Param(KDTreeIndexParams(2)));
//    params.push_back(Param(KDTreeIndexParams(4)));
/*    params.push_back(Param(KDTreeIndexParams(8)));
    params.push_back(Param(KDTreeIndexParams(16)));
    params.push_back(Param(KDTreeIndexParams(32)));

    params.push_back(Param(KMeansIndexParams(8)));
    params.push_back(Param(KMeansIndexParams(16)));
    params.push_back(Param(KMeansIndexParams(32)));
    params.push_back(Param(KMeansIndexParams(64)));
    params.push_back(Param(KMeansIndexParams(128)));
*/
    for(int i = 0; i < trees; ++i) 
      cout << "M" << i << '\t';

    cout << "Dataset\tAlgorithm\tParameter\tRepeat\tRows\tBuild Time\tQPS\tAccuracy" << endl;
    for(FLANNParam& param : params) {
        Dataset& ds = param.getDataset();
        int chunkSize = ds.chunkSize;
        int dim = ds.dim;
        int chunkN = ds.chunkN;
        int initialSize = ds.initialSize;
        int querySize = ds.querySize;
        
        rawDataset = new float[(initialSize + chunkSize * chunkN) * dim];
        readData(ds.path, initialSize + chunkSize * chunkN, dim, rawDataset);
        
        queryDataset = new float[querySize * dim];
        readData(ds.queryPath, querySize, dim, queryDataset);

        for(int r = 0; r < flannRepeat; r++) {
            float *dataset = rawDataset; //shuffle(rawDataset, chunkSize * chunkN + querySize, dim);

            Matrix<float> query(queryDataset, querySize, dim);
            Matrix<float> initial(dataset, initialSize, dim);
            vector<Matrix<float>> chunks;
            Matrix<int> indices(new int[query.rows * nn], query.rows, nn);
            Matrix<float> dists(new float[query.rows * nn], query.rows, nn);
            Matrix<int> answers(new int[query.rows * nn], query.rows, nn);

            for(int i = 0; i < chunkN; ++i) {
                chunks.push_back(Matrix<float>(dataset + initialSize * dim + i * chunkSize * dim, chunkSize, dim));
            }

            Index<L2<float>> index(initial, param.getIndexParams());  
//            index.buildIndex();

            for(int i = 0; i <= chunkN; ++i) {
                // calculate correct answers
                Matrix<float> aggregatedDataset(dataset, initialSize + i * chunkSize, dim);

                timer.begin();
                getCorrectAnswers(aggregatedDataset, initialSize + i * chunkSize, query, answers);
                double correctAnswerTime = timer.end();

                if(param.algorithm() != "KDBalancedTree")
                    for(int j = 0; j < trees; ++j)
                        cout << "0\t";
                // add a new chunk
                timer.begin();
                if(i > 0) {
                    index.addPoints(chunks[i-1]); 
                }
                else
                    index.buildIndex();
                double buildTime = timer.end();

                // do search
                timer.begin();
                index.knnSearch(query, indices, dists, nn, param.getSearchParams());
                double queryTime = timer.end() / querySize;
                double QPS = 1 / queryTime;

                // calculate accuracy
                timer.begin();
                set<int> s;
                int correct = 0;
                for(unsigned int j = 0; j < query.rows; ++j) {
                    s.clear();
                    for(int k = 0; k < nn; ++k) {
                        s.insert(answers[j][k]);
                    }

                    for(int k = 0; k < nn; ++k) {
                        if(s.find(indices[j][k]) != s.end())
                            correct++;
                    }
                }
                double checkTime = timer.end();           
                double accuracy = (double)correct / querySize / nn;

                cout << ds.name << '\t' << param.algorithm() << '\t' << param.format() << '\t' << r << '\t' << initialSize + i * chunkSize << '\t' << buildTime << '\t' << QPS << '\t' << accuracy << endl;
            }

            delete[] indices.ptr();
            delete[] dists.ptr();
            delete[] answers.ptr();
        }

        delete[] rawDataset;
        delete[] queryDataset;
    }

    int chunkSize = 1000;
    int chunkN = 5;
    int querySize = 1000;
    int dim = 100;

    for(auto &trees : annoyTrees) {
        break;
        for(int r = 0; r < annoyRepeat; r++) {
            float *dataset = shuffle(rawDataset, chunkSize * chunkN + querySize, dim);
            AnnoyIndex<int, float, Euclidean, RandRandom> annoyIndex = AnnoyIndex<int, float, Euclidean, RandRandom>(dim);
            Matrix<float> query(dataset, querySize, dim);
            Matrix<int> answers(new int[query.rows * nn], query.rows, nn);

            for(int i = 0; i < chunkN; ++i) {
                // calculate correct answers
                Matrix<float> aggregatedDataset(dataset + querySize * dim, (i + 1) * chunkSize, dim);

                timer.begin();
                getCorrectAnswers(aggregatedDataset, (i + 1) * chunkSize, query, answers);
                double correctAnswerTime = timer.end();

                // add a new chunk
                for(int j = 0; j < chunkSize; ++j ){
                    annoyIndex.add_item(i * chunkSize + j, aggregatedDataset[i * chunkSize + j]); 
                }

                timer.begin();
                //if(((i + 1) & i) == 0) {
                annoyIndex.unbuild();
                annoyIndex.build(trees);
                //}
                double buildTime = timer.end();

                vector<vector<int>> indices(querySize);
                for(int j = 0; j < querySize; ++j) {
                    indices.push_back(vector<int>(nn));
                }

                // do search
                timer.begin();
                
                for(unsigned int j = 0; j < query.rows; ++j) {
                    annoyIndex.get_nns_by_vector(query[j], nn, annoySearchParam, &indices[j], nullptr);
                }
                double queryTime = timer.end() / querySize;
                double QPS = 1 / queryTime;

                // calculate accuracy
                timer.begin();
                set<int> s;
                int correct = 0;
                for(unsigned int j = 0; j < query.rows; ++j) {
                    s.clear();
                    for(int k = 0; k < nn; ++k) {
                        s.insert(answers[j][k]);
                    }

                    for(int k = 0; k < nn; ++k) {
                        if(s.find(indices[j][k]) != s.end())
                            correct++;
                    }
                }
                double checkTime = timer.end();           
                double accuracy = (double)correct / querySize / nn;

                cout << "Annoy" << '\t' << "Annoy(" << trees << ")" << '\t' << r << '\t' << (i + 1) * chunkSize << '\t' << buildTime << '\t' << QPS << '\t' << accuracy << endl;
            }

            delete[] dataset;
            delete[] answers.ptr();
        }
    }

    return 0;
}
