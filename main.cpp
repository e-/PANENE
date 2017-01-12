#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include <iostream>
#include <string>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <memory>

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
            //  cout << shuffled[i * cols + j];
        }
    }
    return shuffled;
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    int chunkSize = 1000;
    int chunkN = 10;
    int querySize = 1000;
    int dim = 100;
    int nn = 10;

    float *rawDataset = new float[(chunkSize * chunkN + querySize) * dim];
    
    readData("data/glove.txt", chunkSize * chunkN + querySize, dim, rawDataset);

    {
        float *dataset = shuffle(rawDataset, chunkSize * chunkN + querySize, dim);

        Matrix<float> query(dataset, querySize, dim);
        vector<Matrix<float>> chunks;
        Matrix<int> indices(new int[query.rows * nn], query.rows, nn);
        Matrix<float> dists(new float[query.rows * nn], query.rows, nn);

        for(int i = 0; i < chunkN; ++i) {
            chunks.push_back(Matrix<float>(dataset + querySize * dim + i * chunkSize * dim, chunkSize, dim));
        }
        
        Index<L2<float>> index(chunks[0], KDTreeIndexParams(4));
        index.buildIndex();

        index.knnSearch(query, indices, dists, nn, flann::SearchParams(128));

/*
        for(int i=0;i<query.rows;++i) {
            for(int j = 0; j< 10; ++j) {
                cout << query[i][j] << ' ';
            }
            cout << endl;
            for(int j=0;j<nn;++j) {
                cout << indices[i][j] << ' '; //dists[i][j] << ' ';
            }
            cout << endl;
        }*/
        delete[] dataset;
        delete[] indices.ptr();
        delete[] dists.ptr();
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
