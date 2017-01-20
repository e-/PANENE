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

int main(int argc, char** argv)
{
    srand(time(NULL));
    int chunkSize = 1000;
    int chunkN = 5;
    int querySize = 1000;
    int dim = 100;
    int nn = 20;
    int flannRepeat = 1;
    int annoyRepeat = 10;

    Timer timer;
    float *rawDataset = new float[(chunkSize * chunkN + querySize) * dim];
    
    int annoySearchParam = 128;
    vector<int> annoyTrees{10, 20, 40, 80, 160};

    vector<Param> params;
    SearchParams searchParam(128);
    searchParam.cores = 1;

    params.push_back(Param(KDTreeIndexParams(8)));
    params.push_back(Param(KDTreeBalancedIndexParams(8, 1.1f)));
    params.push_back(Param(KDTreeBalancedIndexParams(8, 1.1f, FLANN_MEDIAN)));
    params.push_back(Param(KDTreeBalancedIndexParams(8, 100)));

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
    readData("data/glove.txt", chunkSize * chunkN + querySize, dim, rawDataset);

    cout << "Algorithm\tParameter\tRepeat\tRows\tBuild Time\tQPS\tAccuracy" << endl;
  
    for(Param& param : params) {
        for(int r = 0; r < flannRepeat; r++) {
            float *dataset = shuffle(rawDataset, chunkSize * chunkN + querySize, dim);

            Matrix<float> query(dataset, querySize, dim);
            vector<Matrix<float>> chunks;
            Matrix<int> indices(new int[query.rows * nn], query.rows, nn);
            Matrix<float> dists(new float[query.rows * nn], query.rows, nn);
            Matrix<int> answers(new int[query.rows * nn], query.rows, nn);

            for(int i = 0; i < chunkN; ++i) {
                chunks.push_back(Matrix<float>(dataset + querySize * dim + i * chunkSize * dim, chunkSize, dim));
            }

            Index<L2<float>> index(chunks[0], param.getIndexParam());  

            for(int i = 0; i < chunkN; ++i) {
                // calculate correct answers
                Matrix<float> aggregatedDataset(dataset + querySize * dim, (i + 1) * chunkSize, dim);

                timer.begin();
                getCorrectAnswers(aggregatedDataset, (i + 1) * chunkSize, query, answers);
                double correctAnswerTime = timer.end();

                // add a new chunk
                timer.begin();
                if(i > 0) {
                    index.addPoints(chunks[i]);
                }
                else {  
                    index.buildIndex();
                }
                double buildTime = timer.end();

                // do search
                timer.begin();
                index.knnSearch(query, indices, dists, nn, searchParam);
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

                cout << param.algorithm() << '\t' << param.format() << '\t' << r << '\t' << (i + 1) * chunkSize << '\t' << buildTime << '\t' << QPS << '\t' << accuracy << endl;
            }

            delete[] dataset;
            delete[] indices.ptr();
            delete[] dists.ptr();
            delete[] answers.ptr();
        }
    }

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
