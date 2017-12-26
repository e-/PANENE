#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "progressive_tsne.h"

int main(int argv, char *argc[]) {
  // Define some variables
  int origN, N, D, no_dims, max_iter, *landmarks;
  double perc_landmarks;
  double perplexity, theta, *data;
  int rand_seed = -1;

  if(argv <= 1) {
    printf("%s [path_to_input_file]\n", argc[0]);
    return -1;
  }
  
  ProgressiveTSNE* tsne = new ProgressiveTSNE();

  // Read the parameters and the dataset
  if(tsne->load_data(argc[1], &data, &origN, &D, &no_dims, &theta, &perplexity, &rand_seed, &max_iter)) {

    // Make dummy landmarks
    N = origN;
    int* landmarks = (int*) malloc(N * sizeof(int));
    if(landmarks == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    for(int n = 0; n < N; n++) landmarks[n] = n;

    // Now fire up the SNE implementation
    double* Y = (double*) malloc(N * no_dims * sizeof(double));
    double* costs = (double*) calloc(N, sizeof(double));
    if(Y == NULL || costs == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    tsne->run(argc[1], data, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter);

    // Clean up the memory
    free(data); data = NULL;
    free(Y); Y = NULL;
    free(costs); costs = NULL;
    free(landmarks); landmarks = NULL;
  }
  delete(tsne);

  return 0;
}
