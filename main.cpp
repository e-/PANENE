#include <flann/flann.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <memory>
#include "timer.h"
#include <set>

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
    for(int q = 0; q < queryset.rows; q++) {
        int* match = new int[n];
        DistanceType* dists = new DistanceType[n];

        float * query = queryset[q];

        dists[0] = distance(dataset[0], query, dataset.cols);
        match[0] = 0;
        int dcnt = 1;

        for (size_t i=1; i<rows; ++i) {
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

        for (size_t i=0; i<n; ++i) {
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
    int chunkN = 40;
    int querySize = 10000;
    int dim = 100;
    int nn = 10;
    int repeat = 5;
    vector<IndexParams> parameters;

    Timer timer;

    float *rawDataset = new float[(chunkSize * chunkN + querySize) * dim];
    
    parameters.push_back(KMeansIndexParams());
    
    readData("data/glove.txt", chunkSize * chunkN + querySize, dim, rawDataset);

    {
        float *dataset = shuffle(rawDataset, chunkSize * chunkN + querySize, dim);

        Matrix<float> query(dataset, querySize, dim);
        vector<Matrix<float>> chunks;
        Matrix<int> indices(new int[query.rows * nn], query.rows, nn);
        Matrix<float> dists(new float[query.rows * nn], query.rows, nn);
        Matrix<int> answers(new int[query.rows * nn], query.rows, nn);

        for(int i = 0; i < chunkN; ++i) {
            chunks.push_back(Matrix<float>(dataset + querySize * dim + i * chunkSize * dim, chunkSize, dim));
        }
        
        Index<L2<float>> index(chunks[0], KMeansIndexParams()); //KDTreeIndexParams(16)); // AutotunedIndexParams(0.95)); //
        //cout << index.getParameters() << endl;

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
            index.knnSearch(query, indices, dists, nn, SearchParams(128));
            double queryTime = timer.end() / querySize;
            double QPS = 1 / queryTime;

            // calculate accuracy
            timer.begin();
            set<int> s;
            int correct = 0;
            for(int j = 0; j < query.rows; ++j) {
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

            cout << buildTime << " " << QPS << " " << checkTime << " " << correctAnswerTime <<  " " << accuracy << endl;
        }

        delete[] dataset;
        delete[] indices.ptr();
        delete[] dists.ptr();
        delete[] answers.ptr();
    }
    //Matrix<float> dataset;
    //Matrix<float> query;


    /*int nn = 3;
    load_from_file(dataset, "dataset.hdf5","dataset");
    load_from_file(query, "dataset.hdf5","query");

    Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
    Matrix<float> dists(new float[query.rows*nn], query.rows, nn);

    // construct an randomized kd-tree index using 4 kd-trees
    Index<L2<float> > index(dataset, flann::KDTreeIndexParams(4));
    index.buildIndex();                                                                                               

    // do a knn search, using 128 checks
    index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

    flann::save_to_file(indices,"result.hdf5","result");

    delete[] dataset.ptr();
    delete[] query.ptr();
    delete[] indices.ptr();
    delete[] dists.ptr();
    */
    return 0;
}
