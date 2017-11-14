#ifndef panene_kd_tree_h
#define panene_kd_tree_h

namespace panene {

struct InsertionLog
{
  size_t count;
  size_t depth;

  InsertionLog() = default;
  InsertionLog(size_t count_, size_t depth_) : count(count_), depth(depth_) {

  }
};

/***
 * a KD tree that records the frequency and depth of each leaf
 */

template<class NodePtr>
struct KDTree
{
  NodePtr root;
  size_t size;
  size_t capacity;
  int countSum = 0;
  float cost;
  std::vector<InsertionLog> insertionLog;

  KDTree(size_t capacity_) : capacity(capacity_) {
    size = countSum = 0;
    cost = 0;
    root = nullptr;
    insertionLog.resize(capacity);
  }

  ~KDTree() {
    if(root != nullptr)
      root->~Node();
  }

  float computeCost() {
    float cost = 0;

    for (size_t i = 0; i < capacity; ++i) {
      cost += (float)insertionLog[i].count / countSum * insertionLog[i].depth;
    }
    return cost;
  }

  float computeImbalance() {
    float ideal = (float)(log(size) / log(2));
    return computeCost() / ideal;
  }

  size_t computeMaxDepth() {
    size_t maxDepth = 0;
    for (size_t i = 0; i < capacity; ++i) {
      if (maxDepth < insertionLog[i].depth)
        maxDepth = insertionLog[i].depth;
    }
    return maxDepth;
  }

  void setInsertionLog(const size_t id, const size_t count, const size_t depth) {
    countSum = countSum - insertionLog[id].count + count;
    insertionLog[id].count = count;
    insertionLog[id].depth = depth;
  }

  float getCachedCost() {
    return cost;
  }

  float getCachedImbalance() {
    float ideal = (float)(log(size) / log(2));
    return cost / ideal;
  }

  void updateInsertionLog(const size_t id, const size_t count, const size_t depth) {
    if (count > 0 && countSum > 0) {
      auto& prevLog = insertionLog[id];

      cost = (cost
        - (float)prevLog.count / countSum * prevLog.depth)
        * countSum / (countSum - prevLog.count + count)
        + (float)count / (countSum - prevLog.count + count) * depth;
    }

    countSum = countSum - insertionLog[id].count + count;
    insertionLog[id].count = count;
    insertionLog[id].depth = depth;
  }

  void markSplit(const size_t id) {
    size_t count = insertionLog[id].count;
    size_t depth = insertionLog[id].depth;
    updateInsertionLog(id, count + 1, depth + 1);
  }

  std::map<size_t, size_t> computeCountDistribution() {
    std::map<size_t, size_t> dict;

    for (const auto& leaf : insertionLog) {
      if (dict.count(leaf.count) == 0)
        dict[leaf.count] = 0;
      dict[leaf.count]++;
    }

    return dict;
  }
};

}
#endif
