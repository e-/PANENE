#include <progressive_kd_tree_index.h>
#include <data_source/random_data_source.h>
#include <dist.h>

using namespace panene;

typedef size_t IDType;
typedef float ElementType;
using Source = panene::RandomDataSource<IDType, L2<ElementType>>;

int main() {
    const size_t n = 1000;
    const size_t d = 100;
    const size_t k = 20;

    Source randomDataSource(n, d);
    const IndexParams indexParam(4);
    SearchParams searchParam(1024);
    searchParam.cores = 4;
    const float addPointWeight = 0.3f;
    const size_t ops = 100;

    ProgressiveKDTreeIndex<Source> progressiveIndex(&randomDataSource, indexParam, TreeWeight(addPointWeight, 1 - addPointWeight));

    const size_t query_n = 10;
    std::vector<std::vector<ElementType>> queries(query_n);


    for (size_t i = 0; i < query_n; ++i) {
        queries[i].resize(d);

        for (size_t j = 0; j < d; ++j) {
            queries[i][j] = static_cast <ElementType> (rand()) / static_cast <ElementType>(RAND_MAX);
        }
    }

    for (size_t iter = 0; iter < 10; ++iter) {
        std::cout << "Iteration [" << iter << "]" << std::endl;

        progressiveIndex.run(ops);

        std::vector<ResultSet<IDType, ElementType>> results(query_n);
        for (size_t i = 0; i < query_n; ++i) {
            results[i] = ResultSet<IDType, ElementType>(k);
        }

        progressiveIndex.knnSearch(queries, results, k, searchParam);
        for (size_t i = 0; i < 10; ++i) {
            std::cout << results[i] << std::endl;
        }
    }

    return 0;
}
