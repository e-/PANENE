#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include "vptree.h"
#include "sptree.h"
#include "progressive_tsne.h"

using namespace std;
using namespace panene;

// Perform Progressive t-SNE with Progressive KDTree
void ProgressiveTSNE::run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int rand_seed,
    bool skip_random_init, int max_iter, int mom_switch_iter, int print_every) {

  // Set random seed
  if (skip_random_init != true) {
    if(rand_seed >= 0) {
      printf("Using random seed: %d\n", rand_seed);
      srand((unsigned int) rand_seed);
    } else {
      printf("Using current time as random seed...\n");
      srand(time(NULL));
    }
  }

  // Determine whether we are using an exact algorithm
  if(N - 1 < 3 * perplexity) { printf("Perplexity too large for the number of data points!\n"); exit(1); }
  printf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);
  
  if(theta == .0) {
    printf("Exact TSNE is not supported!");
    exit(-1);
  }

  // Set learning parameters
  float total_time = .0;
  clock_t start, end;
  double momentum = .5, final_momentum = .8;
  double eta = 200.0;

  // Allocate some memory
  double* dY    = (double*) malloc(N * no_dims * sizeof(double));
  double* uY    = (double*) malloc(N * no_dims * sizeof(double));
  double* gains = (double*) malloc(N * no_dims * sizeof(double));
  if(dY == NULL || uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;
  for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

  // Initialize data source
  Source source((size_t)N, (size_t)D, X);
 
  // Initialize KNN table

  size_t K = (size_t) (perplexity * 3);
  Sink sink(N, K + 1);

  ProgressiveKNNTable<ProgressiveKDTreeIndex<Source>, Sink> table(
    &source,
    &sink,
    K + 1, 
    IndexParams(1),
    SearchParams(4096),
    TreeWeight(0.3, 0.7),
    TableWeight(0.5, 0.5));

  // Normalize input data (to prevent numerical problems)
  zeroMean(X, N, D);
  double max_X = .0;
  for(int i = 0; i < N * D; i++) {
    if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
  }
  for(int i = 0; i < N * D; i++) X[i] /= max_X;

  // Compute input similarities for exact t-SNE
  double* P = NULL;
  unsigned int* row_P = NULL;
  unsigned int* col_P = NULL;
  double* val_P = NULL;
  double* cur_P = NULL;

  double* normalized_val_P = NULL; // normalized

  // Compute input similarities for approximate t-SNE

  // Initialize solution (randomly)
  if (skip_random_init != true) {
    for(int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
  }

  // Perform main training loop
  start = clock();

  size_t ops = 1000;

  printf("training start\n");
  for(int iter = 0; iter < max_iter; iter++) {
    // Initialize similarity arrays
    initializeSimilarity(N, D, &row_P, &col_P, &val_P, &cur_P, (int) K);

    // Compute asymmetric pairwise input similarities
    computeGaussianPerplexity(&table, ops, X, N, D, row_P, col_P, val_P, cur_P, perplexity, K);

    int n = table.getSize();

    // Symmetrize input similarities
    symmetrizeMatrix(&row_P, &col_P, &val_P, n, N, K);
    
    double sum_P = .0;
    for(int i = 0; i < row_P[n]; i++) sum_P += val_P[i];
    for(int i = 0; i < row_P[n]; i++) val_P[i] /= sum_P;

    // Compute (approximate) gradient
    computeGradient(P, row_P, col_P, val_P, Y, n, no_dims, dY, theta);
  
    // Update gains
    for(int i = 0; i < n * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    for(int i = 0; i < n * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

    // Perform gradient update (with momentum and gains)
    for(int i = 0; i < n * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    for(int i = 0; i < n * no_dims; i++)  Y[i] = Y[i] + uY[i];

    double grad_sum = 0;
    for(int i = 0; i < n * no_dims; i++) {
      grad_sum += dY[i] * dY[i];
    }

    // Make solution zero-mean
    zeroMean(Y, n, no_dims);

    if(iter == mom_switch_iter) momentum = final_momentum;

    // Print out progress
    if (iter > 0 && (iter % print_every == 0 || iter == max_iter - 1)) {
      end = clock();
      double C = .0;
      C = evaluateError(row_P, col_P, val_P, Y, n, no_dims, theta);  // doing approximate computation here!
      if(iter == 0)
        printf("Iteration %d: error is %f\n", iter + 1, C);
      else {
        total_time += (float) (end - start) / CLOCKS_PER_SEC;
        printf("Iteration %d: error is %f (50 iterations in %4.2f seconds) grad_sum is %4.6f\n", iter, C, (float) (end - start) / CLOCKS_PER_SEC, grad_sum);
      }
      start = clock();
    }

    size_t log_every = 1;

    if(iter % log_every == 0) {
      char path[100];
      sprintf(path,"result/result.%d.txt", iter / log_every);

      save_data(path, Y, n, no_dims);
    }
  }
  end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

  // Clean up memory
  free(dY);
  free(uY);
  free(gains);
  free(row_P); row_P = NULL;
  free(col_P); col_P = NULL;
  free(val_P); val_P = NULL;
  
  printf("Fitting performed in %4.2f seconds.\n", total_time);
}


// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void ProgressiveTSNE::computeGradient(double* P, unsigned int* inp_row_P, unsigned int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{

  // Construct space-partitioning tree on current map
  SPTree* tree = new SPTree(D, Y, N);

  // Compute all terms required for t-SNE gradient
  double sum_Q = .0;
  double* pos_f = (double*) calloc(N * D, sizeof(double));
  double* neg_f = (double*) calloc(N * D, sizeof(double));
  if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
  for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

  // Compute final t-SNE gradient
  for(int i = 0; i < N * D; i++) {
    dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
  }
  free(pos_f);
  free(neg_f);
  delete tree;
}

// Compute gradient of the t-SNE cost function (exact)
void ProgressiveTSNE::computeExactGradient(double* P, double* Y, int N, int D, double* dC) {

  // Make sure the current gradient contains zeros
  for(int i = 0; i < N * D; i++) dC[i] = 0.0;

  // Compute the squared Euclidean distance matrix
  double* DD = (double*) malloc(N * N * sizeof(double));
  if(DD == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  computeSquaredEuclideanDistance(Y, N, D, DD);

  // Compute Q-matrix and normalization sum
  double* Q    = (double*) malloc(N * N * sizeof(double));
  if(Q == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  double sum_Q = .0;
  int nN = 0;
  for(int n = 0; n < N; n++) {
    for(int m = 0; m < N; m++) {
      if(n != m) {
        Q[nN + m] = 1 / (1 + DD[nN + m]);
        sum_Q += Q[nN + m];
      }
    }
    nN += N;
  }

  // Perform the computation of the gradient
  nN = 0;
  int nD = 0;
  for(int n = 0; n < N; n++) {
    int mD = 0;
    for(int m = 0; m < N; m++) {
      if(n != m) {
        double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
        for(int d = 0; d < D; d++) {
          dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
        }
      }
      mD += D;
    }
    nN += N;
    nD += D;
  }

  // Free memory
  free(DD); DD = NULL;
  free(Q);  Q  = NULL;
}


// Evaluate t-SNE cost function (approximately)
double ProgressiveTSNE::evaluateError(unsigned int* row_P, unsigned int* col_P, double* val_P, double* Y, int N, int D, double theta)
{

  // Get estimate of normalization term
  SPTree* tree = new SPTree(D, Y, N);
  double* buff = (double*) calloc(D, sizeof(double));
  double sum_Q = .0;
  for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

  // Loop over all edges to compute t-SNE error
  int ind1, ind2;
  double C = .0, Q;
  for(int n = 0; n < N; n++) {
    ind1 = n * D;
    for(int i = row_P[n]; i < row_P[n + 1]; i++) {
      Q = .0;
      ind2 = col_P[i] * D;
      for(int d = 0; d < D; d++) buff[d]  = Y[ind1 + d];
      for(int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
      for(int d = 0; d < D; d++) Q += buff[d] * buff[d];
      Q = (1.0 / (1.0 + Q)) / sum_Q;
      C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
    }
  }

  // Clean up memory
  free(buff);
  delete tree;
  return C;
}

void ProgressiveTSNE::initializeSimilarity(int N, int D, unsigned int** _row_P, unsigned int** _col_P, double** _val_P, double **cur_P, int K)
{
  if(*_row_P != NULL) free(*_row_P);
  if(*_col_P != NULL) free(*_col_P);
  if(*_val_P != NULL) free(*_val_P);

  *_row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
  *_col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
  *_val_P = (double*) calloc(N * K, sizeof(double));
  if(*_row_P == NULL || *_col_P == NULL || *_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  unsigned int* row_P = *_row_P;
  //unsigned int* col_P = *_col_P;
  row_P[0] = 0;
  for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + (unsigned int) K;
  *cur_P = (double*) malloc((N - 1) * sizeof(double));
  if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
}

// Compute input similarities with a fixed perplexity using progressive k-d trees (this function allocates memory another function should free)
// PANENE VERSION
void ProgressiveTSNE::computeGaussianPerplexity(Table *table, size_t ops, double* X, int N, int D, unsigned int* row_P, unsigned int* col_P, double* val_P, double* cur_P, double perplexity, int K) {

  if(perplexity > K) printf("Perplexity should be lower than K!\n");

  // Update the KNNTable

  //printf("Updating the KNN table...\n");
  UpdateResult ar = table->run(ops);
  //printf("Done! %d points are inserted, %d rows are updated\n", ar.addPointResult, ar.updateTableResult);

  // Build ball tree on data set
  //VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
  //vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
  //for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
  //tree->create(obj_X);

  // Loop over all points to find nearest neighbors
  //vector<DataPoint> indices;
  //vector<double> distances;
  
  /*
    We need to compute val_P for points that are
      1) newly inserted (ar.addPointResult points)
      2) updated (ar.updatePointResult points)

    ar.updatedIds has the ids of the updated points and the ids of the newly added points can be computed by comparing table.getSize() and ar.addPointResult
   */
  
  // collect all ids that need to be updated
  
  size_t s = table->getSize();
  for(int i = 0; i < s; ++i)
  {
    // Find nearest neighbors
    const size_t *indices = table->getNeighbors(i);
    const double *distances = table->getDistances(i);
    //indices.clear();
    //distances.clear();
    //tree->search(obj_X[n], K + 1, &indices, &distances);

    // Initialize some variables for binary search
    bool found = false;
    double beta = 1.0;
    double min_beta = -DBL_MAX;
    double max_beta =  DBL_MAX;
    double tol = 1e-5;

    // Iterate until we found a good perplexity
    int iter = 0; double sum_P;
    while(!found && iter < 200) {

      // Compute Gaussian kernel row
      for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

      // Compute entropy of current row
      sum_P = DBL_MIN;
      for(int m = 0; m < K; m++) sum_P += cur_P[m];
      double H = .0;
      for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
      H = (H / sum_P) + log(sum_P);

      // Evaluate whether the entropy is within the tolerance level
      double Hdiff = H - log(perplexity);
      if(Hdiff < tol && -Hdiff < tol) {
        found = true;
      }
      else {
        if(Hdiff > 0) {
          min_beta = beta;
          if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
            beta *= 2.0;
          else
            beta = (beta + max_beta) / 2.0;
        }
        else {
          max_beta = beta;
          if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
            beta /= 2.0;
          else
            beta = (beta + min_beta) / 2.0;
        }
      }

      // Update iteration counter
      iter++;
    }

    // Row-normalize current row of P and store in matrix
    for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
    for(unsigned int m = 0; m < K; m++) {
      col_P[row_P[i] + m] = (unsigned int) indices[m + 1]; //.index();
      val_P[row_P[i] + m] = cur_P[m];
    }
  }

  // Clean up memory
  //obj_X.clear();
  free(cur_P);
  //delete tree;
}


// Symmetrizes a sparse matrix
void ProgressiveTSNE::symmetrizeMatrix(unsigned int** _row_P, unsigned int** _col_P, double** _val_P, int effectiveN, int N, int K) {

  // Get sparse matrix
  unsigned int* row_P = *_row_P;
  unsigned int* col_P = *_col_P;
  double* val_P = *_val_P;

  // Count number of elements and row counts of symmetric matrix
  int* row_counts = (int*) calloc(N, sizeof(int));
  if(row_counts == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  for(int n = 0; n < effectiveN; n++) {
    for(int i = row_P[n]; i < row_P[n + 1]; i++) {
      if(col_P[i] >= N) { // removed neighbors
        row_counts[n]++;
        continue;
      } 

      // Check whether element (col_P[i], n) is present
      bool present = false;
      for(int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
        if(col_P[m] == n) present = true;
      }
      if(present) row_counts[n]++;
      else {
        row_counts[n]++;
        row_counts[col_P[i]]++;
      }
    }
  }
  for(int n = effectiveN; n < N; ++n)
    row_counts[n] = K;

  int no_elem = 0;
  for(int n = 0; n < N; n++) no_elem += row_counts[n];

  // Allocate memory for symmetrized matrix
  unsigned int* sym_row_P = (unsigned int*) malloc((N + 1) * sizeof(unsigned int));
  unsigned int* sym_col_P = (unsigned int*) malloc(no_elem * sizeof(unsigned int));
  for(int i=0;i<no_elem;++i) sym_col_P[i] = N + 1;
  double* sym_val_P = (double*) malloc(no_elem * sizeof(double));
  if(sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }

  // Construct new row indices for symmetric matrix
  sym_row_P[0] = 0;
  for(int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + (unsigned int) row_counts[n];

  // Fill the result matrix
  int* offset = (int*) calloc(N, sizeof(int));
  if(offset == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  for(int n = 0; n < effectiveN; n++) {
    for(unsigned int i = row_P[n]; i < row_P[n + 1]; i++) {                                  // considering element(n, col_P[i])
      if(col_P[i] >= N) { // removed neighbors
        continue;
      } 
      // Check whether element (col_P[i], n) is present
      bool present = false;
      for(unsigned int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
        if(col_P[m] == n) {
          present = true;
          if(n <= col_P[i]) {                                                 // make sure we do not add elements twice
            sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
            sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
            sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
            sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
          }
        }
      }

      // If (col_P[i], n) is not present, there is no addition involved
      if(!present) {
        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
      }

      // Update offsets
      if(!present || (present && n <= col_P[i])) {
        offset[n]++;
        if(col_P[i] != n) offset[col_P[i]]++;
      }
    }
  }

  // Divide the result by two
  for(int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

  // Return symmetrized matrices
  free(*_row_P); *_row_P = sym_row_P;
  free(*_col_P); *_col_P = sym_col_P;
  free(*_val_P); *_val_P = sym_val_P;

  // Free up some memery
  free(offset); offset = NULL;
  free(row_counts); row_counts = NULL;
}

// Compute squared Euclidean distance matrix
void ProgressiveTSNE::computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
  const double* XnD = X;
  for(int n = 0; n < N; ++n, XnD += D) {
    const double* XmD = XnD + D;
    double* curr_elem = &DD[n*N + n];
    *curr_elem = 0.0;
    double* curr_elem_sym = curr_elem + N;
    for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
      *(++curr_elem) = 0.0;
      for(int d = 0; d < D; ++d) {
        *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
      }
      *curr_elem_sym = *curr_elem;
    }
  }
}


// Makes data zero-mean
void ProgressiveTSNE::zeroMean(double* X, int N, int D) {
  // Compute data mean
  double* mean = (double*) calloc(D, sizeof(double));
  if(mean == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  int nD = 0;
  for(int n = 0; n < N; n++) {
    for(int d = 0; d < D; d++) {
      mean[d] += X[nD + d];
    }
    nD += D;
  }
  for(int d = 0; d < D; d++) {
    mean[d] /= (double) N;
  }

  // Subtract data mean
  nD = 0;
  for(int n = 0; n < N; n++) {
    for(int d = 0; d < D; d++) {
      X[nD + d] -= mean[d];
    }
    nD += D;
  }
  free(mean); mean = NULL;
}


// Generates a Gaussian random number
double ProgressiveTSNE::randn() {
  double x, y, radius;
  do {
    x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    radius = (x * x) + (y * y);
  } while((radius >= 1.0) || (radius == 0.0));
  radius = sqrt(-2 * log(radius) / radius);
  x *= radius;
  y *= radius;
  return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool ProgressiveTSNE::load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter) {

  // Open file, read first 2 integers, allocate memory, and read the data
  FILE *h;
  if((h = fopen("data.txt", "r")) == NULL) {
    printf("Error: could not open data file.\n");
    return false;
  }

  fscanf(h, "%d", n); // fread(n, sizeof(int), 1, h);											// number of datapoints
  fscanf(h, "%d", d); // fread(d, sizeof(int), 1, h);											// original dimensionality
  fscanf(h, "%lf", theta); // fread(theta, sizeof(double), 1, h);										// gradient accuracy
  fscanf(h, "%lf", perplexity); // fread(perplexity, sizeof(double), 1, h);								// perplexity
  fscanf(h, "%d", no_dims); //	fread(no_dims, sizeof(int), 1, h);                                      // output dimensionality
  fscanf(h, "%d", max_iter); // fread(max_iter, sizeof(int),1,h);                                       // maximum number of iterations

  *data = (double*) malloc(*d * *n * sizeof(double));
  if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }

  for(int i = 0; i < *n; ++i)
    for(int j = 0; j < *d; ++j)
      fscanf(h, "%lf", (*data+i*(*d)+j));
  //    fread(*data, sizeof(double), *n * *d, h);                               // the data

  if(!feof(h)) fread(rand_seed, sizeof(int), 1, h);                       // random seed
  fclose(h);
  printf("Read the %i x %i data matrix successfully!\n", *n, *d);
  return true;
}

// Function that saves map to a t-SNE file
void ProgressiveTSNE::save_data(char *path, double* data, int n, int d) {
  FILE *h;
  if((h = fopen(path, "w")) == NULL) {
    printf("Error: could not open data file.\n");
    return;
  }
  
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < d; ++j)
      fprintf(h, "%lf ", data[i*d+j]);
    fprintf(h, "\n");
  }
  
  fclose(h);
  printf("Wrote the %i x %i data matrix successfully!\n", n, d);
}
