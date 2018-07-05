/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef RESPONSIVE_TSNE_H
#define RESPONSIVE_TSNE_H

#include <progressive_knn_table.h>
#include <data_source/array_data_source.h>
#include <data_sink/vector_data_sink.h>
#include <dist.h>
#include <vector>
#include <map>

#include "../lib/config.h"

using namespace panene;
using namespace std;

typedef size_t IDType;
typedef double ElementType;
using Source = ArrayDataSource<IDType, L2<ElementType>>;
typedef typename Source::DistanceType DistanceType;
using Sink = VectorDataSink<IDType, DistanceType>;
using Table = ProgressiveKNNTable<ProgressiveKDTreeIndex<Source>, Sink>;

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

class ResponsiveTSNE
{
public:
    void run(double* X, size_t N, size_t D, double* Y, size_t no_dims, double perplexity, double theta, int rand_seed,
             bool skip_random_init, size_t max_iter, size_t stop_lying_iter, size_t mom_switch_iter, Config& config);
    void updateSimilarity(Table *table, 
      vector<map<size_t, double>>& neighbors,
      vector<map<size_t, double>>& similarties,
      double* Y,
      size_t no_dims,
      double perplexity,
      size_t K,
      size_t ops,
      float ee_factor);

private:
    void computeGradient(vector<map<size_t, double>>& similarities, double* Y, size_t N, size_t D, double* dC, double theta, float ee_factor);
    double evaluateError(vector<map<size_t, double>>& similarities, double* Y, size_t N, size_t D, double theta, float ee_factor);
    void zeroMean(double* X, size_t N, size_t D);
    double randn();
};

#endif
