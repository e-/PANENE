#include <benchmark/metadata.h>
#include <data_sink/vector_data_sink.h>
#include <kd_tree_index.h>
#include <dist.h>
#include <data_source/random_data_source.h>

using namespace panene;

typedef size_t IDType;
typedef float ElementType;
using Source = RandomDataSource<IDType, L2<ElementType>>;
typedef L2<ElementType>::ResultType DistanceType;

void run() {
    size_t n = 10000000;
    size_t dim = 100;
    size_t tries = 30;

    Source dataSource(n, dim);    
    const IndexParams indexParam(1);
    Timer timer;

    dataSource.generate();
    KDTreeIndex<Source> index(&dataSource, indexParam);

    double elapsed = 0;

    for (size_t i = 0; i < tries; i++) {
        std::vector<IDType> ids(n);
        for (size_t j = 0; j < n; j++) ids[j] = j;

        timer.begin();
        int idx;
        int cutfeat;
        DistanceType cutval;
        index.meanSplit(&ids[0], n, idx, cutfeat, cutval);
        elapsed += timer.end();
    }
    
    std::cout << "n = " << n << ", dim = " << dim << std::endl;
    std::cout << "avg time: " << elapsed / tries << std::endl;

    std::cin >> n;
}

int main(int argc, const char **argv) {
    run();

    return 0;
}
