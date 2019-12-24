#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "tsne.h"
#include "../lib/config.h"

// Function that runs the Barnes-Hut implementation of t-SNE
int main(int argv, char *argc[]) {

    // Define some variables
    TSNE* tsne = new TSNE();
    std::string path = "D:\\G\\work\\panene\\panene\\examples\\tsne\\config.txt";
    if (argv >= 2)
        path = argc[1];

    Config config = Config::load(path);

    // Make dummy landmarks
    int* landmarks = (int*)malloc(config.n * sizeof(int));
    if (landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for (int n = 0; n < config.n; n++) landmarks[n] = n;

    // Now fire up the SNE implementation
    double* Y = (double*)malloc(config.n * config.output_dims * sizeof(double));
    double* costs = (double*)calloc(config.n, sizeof(double));
    if (Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tsne->run(&config.data[0], config.n, config.input_dims, Y, config.output_dims,
        config.perplexity, config.theta, config.seed, false, config.max_iter, config.ee_iter, config.ee_iter, config);

    config.save(Y);

    // Clean up the memory
    free(Y); Y = NULL;
    free(costs); costs = NULL;
    free(landmarks); landmarks = NULL;

    delete(tsne);
}
