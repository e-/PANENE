#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <random>
#include <chrono>

#include "config.h"
#include "vptree.h"
#include "sptree.h"
#include "progressive_tsne.h"

using namespace std;
using namespace panene;

float getEEFactor(int iter, int stop_lying_factor) {
#if USE_EE

#if PERIODIC_EE
  if(iter % PERIODIC_EE_CYCLE < PERIODIC_EE_DURATION) return EE_FACTOR;
  return 1.0f;
#else

  if(iter < stop_lying_factor) return EE_FACTOR;
  return 1.0f;
#endif

#else
  return 1.0f;
#endif

}

vector<size_t> getRandomBatch(size_t n, size_t m) { // get m from [0, n)
  vector<size_t> indices(n);
  for(size_t i=0;i<n;++i) indices[i]=i;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

  shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

  return vector<size_t>(indices.begin(), indices.begin() + std::min(n, m));
}

// Perform Progressive t-SNE with Progressive KDTree
void ProgressiveTSNE::run(char *path, double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int rand_seed,
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
  printf("Using no_dims = %d, perplexity = %f, theta = %f, USE_ADAM=%d, EE_FACTOR=%4f\n", no_dims, perplexity, theta, USE_ADAM, EE_FACTOR);

  if(theta == .0) {
    printf("Exact TSNE is not supported!");
    exit(-1);
  }

  // Set learning parameters
  float total_time = .0;
  clock_t start, end;
  double momentum = .5, final_momentum = .8;

  double* dY    = (double*) malloc(N * no_dims * sizeof(double));
  if(dY == NULL) { printf("Memory allocation failed!\n"); exit(1); }

#if USE_ADAM
  // Adam optimzer
  double* m_t    = (double*) malloc(N * no_dims * sizeof(double));
  double* v_t    = (double*) malloc(N * no_dims * sizeof(double));
  if(m_t == NULL || v_t == NULL) { printf("Memory allocation failed!\n"); exit(1); }

  for(int i = 0; i < N * no_dims; i++) m_t[i] = v_t[i] = .0;

  double eta = 1;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double pbeta1 = beta1;
  double pbeta2 = beta2;
  double eps = 1e-8;

#else
  // Allocate some memory
  double eta = 200;
  double* uY    = (double*) malloc(N * no_dims * sizeof(double));
  double* gains = (double*) malloc(N * no_dims * sizeof(double));
  if(uY == NULL || gains == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;
  for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;
#endif

  // Create a result directory
  char base_path[200];
  sprintf(base_path, "%s.progressive", path);

  struct stat st = {0};
  if(stat(base_path, &st) == -1) {
    mkdir(base_path, 0700);
  } 
  
  // Logging
  char meta_path[200];
  sprintf(meta_path, "%s/metadata.txt", base_path);
  FILE *meta = fopen(meta_path, "w");
  
  // Print parameters
  fprintf(meta, "path=%s, N=%d, D=%d, no_dims=%d, perplexity=%3f, theta=%3f, max_iter=%d, mom_switch_iter=%d\n", path, N, D, no_dims, perplexity, theta, max_iter, mom_switch_iter);
  fprintf(meta, "eta=%3lf, USE_EE=%d, EE_FACTOR=%3f, PERIODIC_EE=%d, PERIODIC_EE_CYCLE=%d, PERIODIC_EE_DURATION=%d, PERIODIC_RESET=%d, BATCH_SIZE=%d\n", eta, USE_EE, EE_FACTOR, PERIODIC_EE, PERIODIC_EE_CYCLE, PERIODIC_EE_DURATION, PERIODIC_RESET, BATCH_SIZE);

  start = clock();

  // Initialize data source
  Source source((size_t)N, (size_t)D, X);

  // Initialize KNN table
  size_t K = (size_t) (perplexity * 3);
  Sink sink(N, K + 1);

  ProgressiveKNNTable<ProgressiveKDTreeIndex<Source>, Sink> table(
      &source,
      &sink,
      K + 1, 
      IndexParams(4),
      SearchParams(1024),
      TreeWeight(0.7, 0.3),
      TableWeight(0.5, 0.5));

  // Normalize input data (to prevent numerical problems)
  zeroMean(X, N, D);
  double max_X = .0;
  for(int i = 0; i < N * D; i++) {
    if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
  }
  for(int i = 0; i < N * D; i++) X[i] /= max_X;

  // Initialize solution (randomly)

  if (skip_random_init != true) {
    srand(0);
    for(int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
  }

  end = clock();

  total_time += (float)(end - start) / CLOCKS_PER_SEC;
  printf("took %f for setup\n", total_time);
  fprintf(meta, "embedding %f\n", total_time);

  // Perform main training loop

  size_t ops = 300;

  vector<map<size_t, double>> neighbors(N); // neighbors[i] has exact K items
  vector<map<size_t, double>> similarities(N); // may have more due to asymmetricity of neighborhoodship 

  printf("training start\n");
  float old_ee_factor = 1.0f;
  for(int iter = 0; iter < max_iter; iter++) {
    start = clock();

    float start_perplex = 0, end_perplex = 0;
    float table_time = 0;
    float ee_factor = getEEFactor(iter, 100);

    if(old_ee_factor != ee_factor) {
      float ratio = ee_factor / old_ee_factor;
      printf("EE changed, ratio = %3f\n", ratio);
      for(auto &sim : similarities) {
        for(auto &kv : sim ) {
          kv.second *= ratio;
        }
      }
#if PERIODIC_EE && PERIODIC_RESET
      if(ee_factor == EE_FACTOR) {      
        for(int i = 0; i < N * no_dims; i++)    uY[i] =  .0;
        for(int i = 0; i < N * no_dims; i++) gains[i] = 1.0;
      }
#endif
    }

    old_ee_factor = ee_factor;
    printf("ee_factor = %.3f\n", ee_factor);

    if(table.getSize() < N) {
      start_perplex = clock();
      updateSimilarity(&table, neighbors, similarities, Y, no_dims, perplexity, K, ops, ee_factor);
      end_perplex = clock();
    }

    int n = table.getSize();

    // Get a random batch
    vector<size_t> batch = getRandomBatch(n, BATCH_SIZE);
    
    // Compute (approximate) gradient for the batch
    // computeGradient(batch, similarities, Y, n, no_dims, dY, theta, ee_factor);

#if USE_ADAM
    // we use Adam Optimzer
    for(int i = 0; i < n * no_dims; i++) {
      m_t[i] = beta1 * m_t[i] + (1 - beta1) * dY[i];
      v_t[i] = beta2 * v_t[i] + (1 - beta2) * dY[i] * dY[i];
      double d = eta * sqrt(1 - pbeta2) / (1 - pbeta1) / (sqrt(v_t[i]) + eps) * m_t[i];
      Y[i] = Y[i] - d;
    }

    pbeta1 *= beta1;
    pbeta2 *= beta2;
#else

    // Update gains
    for(int i = 0; i < n * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    for(int i = 0; i < n * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

    // Perform gradient update (with momentum and gains)
    for(int i = 0; i < n * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    for(int i = 0; i < n * no_dims; i++)  Y[i] = Y[i] + uY[i];

#endif

    double grad_sum = 0;
    for(int i = 0; i < n * no_dims; i++) {
      grad_sum += dY[i] * dY[i];
    }

    // Make solution zero-mean
    zeroMean(Y, n, no_dims);

    // if(iter == mom_switch_iter) momentum = final_momentum;

    end = clock();
    total_time += (float) (end - start) / CLOCKS_PER_SEC;

    // Print out progress
    size_t log_every = 5;

    if(iter < 20 || iter % log_every == 0) {
      double C = .0;
      C = evaluateError(similarities, Y, N, no_dims, theta, ee_factor);  // doing approximate computation here!
      printf("Iteration %d: error is %f (total=%4.2f seconds, perplex=%4.2f seconds, tree=%4.2f seconds, ee_factor=%2.1f) grad_sum is %4.6f\n", iter, C, (float) (end - start) / CLOCKS_PER_SEC, (end_perplex - start_perplex) / CLOCKS_PER_SEC, table_time, ee_factor, grad_sum);

      char path[200];
      sprintf(path, "%s/result.%d.txt", base_path, iter);
      save_data(path, Y, n, no_dims);
      fprintf(meta, "%d %f %lf %s/result.%d.txt\n", iter, total_time, C, base_path, iter);
    }
  }
  // Clean up memory
  free(dY);

#if USE_ADAM
  free(m_t);
  free(v_t);
#else
  free(uY);
  free(gains);
#endif


  fclose(meta);

  printf("Fitting performed in %4.2f seconds.\n", total_time);

  // Save the final embedding
  char emb_path[200];
  sprintf(emb_path, "%s/embedding.txt", base_path);
  save_data(emb_path, Y, N, no_dims);
}

void ProgressiveTSNE::updateSimilarity(Table *table, 
    vector<map<size_t, double>>& neighbors,
    vector<map<size_t, double>>& similarities,
    double* Y,
    size_t no_dims,
    double perplexity,
    size_t K,
    size_t ops,
    float ee_factor) {

  if(perplexity > K) printf("Perplexity should be lower than K!\n");

  // Update the KNNTable
  clock_t start = clock();
  UpdateResult ar = table->run(ops);
  clock_t end = clock();

  float table_time = (end - start) / CLOCKS_PER_SEC;
  /*
     We need to compute val_P for points that are
     1) newly inserted (points in ar.addPointResult)
     2) updated (points in ar.updatePointResult)

     ar.updatedIds has the ids of the updated points. 
     The ids of the newly added points can be computed by comparing table.getSize() and ar.addPointResult
     */

  // collect all ids that need to be updated
  vector<size_t> updated;
  vector<double> p(K);
  map<size_t, map<size_t, double>> old;

  for(size_t i = table->getSize() - ar.addPointResult; i < table->getSize(); ++i) {
    // point i has been newly inserted. 
    ar.updatedIds.insert(i);

    // for newly added points, we set its initial position to the mean of its neighbors
    const size_t *indices = table->getNeighbors(i);

    for(size_t j = i * no_dims; j < (i + 1) * no_dims; ++j) {
      Y[j] = 0;
    }

    for(size_t k = 0; k < K; ++k) {
      for(size_t j = 0; j < no_dims; ++j) {
        Y[i * no_dims + j] += Y[indices[k + 1] * no_dims + j] / K;
      }
    }
  }

  for(size_t uid : ar.updatedIds) { 
    // the neighbors of uid has been updated
    const size_t *indices = table->getNeighbors(uid);
    const double *distances = table->getDistances(uid);   

    bool found = false;
    double beta = 1.0;
    double min_beta = -DBL_MAX;
    double max_beta =  DBL_MAX;
    double tol = 1e-5;

    int iter = 0; 
    double sum_P;

    // Iterate until we found a good perplexity
    while(!found && iter < 200) {

      // Compute Gaussian kernel row
      for(int m = 0; m < K; m++) p[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

      // Compute entropy of current row
      sum_P = DBL_MIN;
      for(int m = 0; m < K; m++) sum_P += p[m];
      
      double H = .0;
      for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * p[m]);
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

    for(unsigned int m = 0; m < K; m++) p[m] /= sum_P;

    old[uid] = neighbors[uid];
    neighbors[uid].clear();
    for(unsigned int m = 0; m < K; m++) {
      neighbors[uid][indices[m+1]] = p[m] * ee_factor;
    }
  } 

  for(size_t Aid : ar.updatedIds) { // point A
    // neighbors changed, we need to keep the similarities symmetric

    // the neighbors of uid has been updated
    for(auto &it: neighbors[Aid]) { // point B
      size_t Bid = it.first;
      
      double sAB = it.second;
      double sBA = 0;

      if(neighbors[Bid].count(Aid) > 0) 
        sBA = neighbors[Bid][Aid];

      similarities[Aid][Bid] = (sAB + sBA) / 2;
      similarities[Bid][Aid] = (sAB + sBA) / 2;
    }

    for(auto &it: old[Aid]) {
      size_t oldBid = it.first;

      if(neighbors[Aid].count(oldBid) > 0) continue;

      // exit points

      double sAB = 0;
      double sBA = 0;

      if(neighbors[oldBid].count(Aid) > 0)
        sBA = neighbors[oldBid][Aid];

      if(sBA == 0) {
        similarities[Aid].erase(oldBid);
        similarities[oldBid].erase(Aid);
      }
      else {
        similarities[Aid][oldBid] = (sAB + sBA) / 2;
        similarities[oldBid][Aid] = (sAB + sBA) / 2;
      }
    }
  }
  
  //for(auto &it: similarities[0]) {
    //printf("[%d] %lf\n", it.first, it.second);
  //}
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void ProgressiveTSNE::computeGradient(vector<size_t>& batch, vector<map<size_t, double>>& similarities, double* Y, int N, int D, double* dC, double theta, float ee_factor)
{
  // Construct space-partitioning tree on current map
  SPTree* tree = new SPTree(D, Y, N);

  // Compute all terms required for t-SNE gradient
  double sum_Q = .0;
  double* pos_f = (double*) calloc(N * D, sizeof(double));
  double* neg_f = (double*) calloc(N * D, sizeof(double));
  if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  tree->computeEdgeForces(similarities, N, pos_f, ee_factor);
  for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

  // Compute final t-SNE gradient
  for(int i = 0; i < N * D; i++) {
    dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
  }
  free(pos_f);
  free(neg_f);
  delete tree;
}

// Evaluate t-SNE cost function (approximately)
double ProgressiveTSNE::evaluateError(vector<map<size_t, double>>& similarities,
   double *Y, int N, int D, double theta, float ee_factor)
{

  // Get estimate of normalization term
  SPTree* tree = new SPTree(D, Y, N);
  double* buff = (double*) calloc(D, sizeof(double));
  double sum_Q = .0;
  for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, buff, &sum_Q);

  // Loop over all edges to compute t-SNE error
  int ind1 = 0, ind2;
  double C = .0, Q;

  double sum_P = .0;
  int j = 0;
  for(auto &p : similarities) {
    if(j >= N) break;
    j++;
    for(auto &it: p) {
      sum_P += it.second;
    }
  }

  sum_P /= ee_factor;

  j = 0;
  for(auto &p : similarities) {
    if(j >= N) break;
    for(auto &it: p) {
      Q = .0;
      ind2 = it.first * D;
      for(int d = 0; d < D; d++) buff[d]  = Y[ind1 + d];
      for(int d = 0; d < D; d++) buff[d] -= Y[ind2 + d];
      for(int d = 0; d < D; d++) Q += buff[d] * buff[d];
      Q = (1.0 / (1.0 + Q)) / sum_Q;
      C += it.second / sum_P * log((it.second / sum_P + FLT_MIN) / (Q + FLT_MIN));
    }
    ind1 += D;
    j++;
  }

  // Clean up memory
  free(buff);
  delete tree;
  return C;
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
bool ProgressiveTSNE::load_data(char *path, double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter) {

  // Open file, read first 2 integers, allocate memory, and read the data
  FILE *h;
  if((h = fopen(path, "r")) == NULL) {
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
