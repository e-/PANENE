#ifndef PAKNN_RESULT_SET_H_
#define PAKNN_RESULT_SET_H_

#include <cstdio>
#include <limits>
#include <vector>

namespace panene {

template <typename DistanceType>
struct DistanceIndex
{
  DistanceIndex(DistanceType dist, size_t index) :
    dist_(dist), index_(index), tree_(0)
  {

  }

  DistanceIndex(DistanceType dist, size_t index, int tree) :
    dist_(dist), index_(index), tree_(tree)
  {

  }

  bool operator<(const DistanceIndex& dist_index) const
  {
    return (dist_ < dist_index.dist_) || ((dist_ == dist_index.dist_) && index_ < dist_index.index_);

  }
  DistanceType dist_;
  size_t index_;
  int tree_;

};

template <typename DistanceType>
class ResultSet
{
public:
    virtual ~ResultSet() {}

    virtual bool full() const = 0;

    virtual void addPoint(DistanceType dist, size_t index, int tree = 0) = 0;

    virtual DistanceType worstDist() const = 0;
    
    virtual void getTrees(std::vector<int> &result) {};
};

/**
 * KNNSimpleResultSet does not ensure that the element it holds are unique.
 * Is used in those cases where the nearest neighbour algorithm used does not
 * attempt to insert the same element multiple times.
 */
template <typename DistanceType>
class KNNSimpleResultSet : public ResultSet<DistanceType>
{
public:
	typedef DistanceIndex<DistanceType> DistIndex;

	KNNSimpleResultSet(size_t capacity_) :
        capacity_(capacity_)
    {
		// reserving capacity to prevent memory re-allocations
		dist_index_.resize(capacity_, DistIndex(std::numeric_limits<DistanceType>::max(),-1));
    	clear();
    }

    ~KNNSimpleResultSet()
    {
    }

    /**
     * Clears the result set
     */
    void clear()
    {
        worst_distance_ = std::numeric_limits<DistanceType>::max();
        dist_index_[capacity_-1].dist_ = worst_distance_;
        count_ = 0;
    }

    /**
     *
     * @return Number of elements in the result set
     */
    size_t size() const
    {
        return count_;
    }

    /**
     * Radius search result set always reports full
     * @return
     */
    bool full() const
    {
        return count_==capacity_;
    }

    /**
     * Add a point to result set
     * @param dist distance to point
     * @param index index of point
     */
    void addPoint(DistanceType dist, size_t index, int tree = 0)
    {
    	if (dist>=worst_distance_) return;

        if (count_ < capacity_) ++count_;
        size_t i;
        for (i=count_-1; i>0; --i) {
#ifdef FLANN_FIRST_MATCH
            if ( (dist_index_[i-1].dist_>dist) || ((dist==dist_index_[i-1].dist_)&&(dist_index_[i-1].index_>index)) )
#else
            if (dist_index_[i-1].dist_>dist)
#endif
            {
            	dist_index_[i] = dist_index_[i-1];
            }
            else break;
        }
        dist_index_[i].dist_ = dist;
        dist_index_[i].index_ = index;
        dist_index_[i].tree_ = tree;
        worst_distance_ = dist_index_[capacity_-1].dist_;
    }

    /**
     * Copy indices and distances to output buffers
     * @param indices
     * @param dists
     * @param num_elements Number of elements to copy
     * @param sorted Indicates if results should be sorted
     */
    void copy(size_t* indices, DistanceType* dists, size_t num_elements, bool sorted = true)
    {
    	size_t n = std::min(count_, num_elements);
    	for (size_t i=0; i<n; ++i) {
    		*indices++ = dist_index_[i].index_;
    		*dists++ = dist_index_[i].dist_;
    	}
    }

    DistanceType worstDist() const
    {
    	return worst_distance_;
    }

    void getTrees(std::vector<int> &result) {
      size_t n = dist_index_.size();
      for(size_t i = 0; i < n; ++i) {
          result.push_back(dist_index_[i].tree_);
      }
    };
private:
    size_t capacity_;
    size_t count_;
    DistanceType worst_distance_;
    std::vector<DistIndex> dist_index_;
};

/**
 * K-Nearest neighbour result set. Ensures that the elements inserted are unique
 */
template <typename DistanceType>
class KNNResultSet : public ResultSet<DistanceType>
{
public:
	typedef DistanceIndex<DistanceType> DistIndex;

    KNNResultSet(int capacity) : capacity_(capacity)
    {
		// reserving capacity to prevent memory re-allocations
			dist_index_.resize(capacity_, DistIndex(std::numeric_limits<DistanceType>::max(),-1));
    	clear();
    }

    ~KNNResultSet()
    {

    }

    /**
     * Clears the result set
     */
    void clear()
    {
        worst_distance_ = std::numeric_limits<DistanceType>::max();
        dist_index_[capacity_-1].dist_ = worst_distance_;
        count_ = 0;
    }

    size_t size() const
    {
        return count_;
    }

    bool full() const
    {
        return count_ == capacity_;
    }


    void addPoint(DistanceType dist, size_t index, int tree = 0)
    {
        if (dist >= worst_distance_) return;
        size_t i;
        for (i = count_; i > 0; --i) {
#ifdef FLANN_FIRST_MATCH
            if ( (dist_index_[i-1].dist_<=dist) && ((dist!=dist_index_[i-1].dist_)||(dist_index_[i-1].index_<=index)) )
#else
            if (dist_index_[i-1].dist_<=dist)
#endif
            {
                // Check for duplicate indices
                for (size_t j = i - 1; dist_index_[j].dist_ == dist && j--;) {
                    if (dist_index_[j].index_ == index) {
                        return;
                    }
                }
                break;
            }
        }
        if (count_ < capacity_) ++count_;
        for (size_t j = count_-1; j > i; --j) {
            dist_index_[j] = dist_index_[j-1];
        }
        dist_index_[i].dist_ = dist;
        dist_index_[i].index_ = index;
        dist_index_[i].tree_ = tree;
        worst_distance_ = dist_index_[capacity_-1].dist_;
    }

    /**
     * Copy indices and distances to output buffers
     * @param indices
     * @param dists
     * @param num_elements Number of elements to copy
     * @param sorted Indicates if results should be sorted
     */
    void copy(size_t* indices, DistanceType* dists, size_t num_elements, bool sorted = true)
    {
    	size_t n = std::min(count_, num_elements);
    	for (size_t i=0; i<n; ++i) {
    		*indices++ = dist_index_[i].index_;
    		*dists++ = dist_index_[i].dist_;
    	}
    }

    DistanceType worstDist() const
    {
        return worst_distance_;
    }

private:
    size_t capacity_;
    size_t count_;
    DistanceType worst_distance_;
    std::vector<DistIndex> dist_index_;
};


template <typename DistanceType>
class KNNResultSet2 : public ResultSet<DistanceType>
{
public:
	typedef DistanceIndex<DistanceType> DistIndex;

	KNNResultSet2(size_t capacity_) :
        capacity_(capacity_)
    {
		// reserving capacity to prevent memory re-allocations
		dist_index_.reserve(capacity_);
    	clear();
    }

    ~KNNResultSet2()
    {
    }

    /**
     * Clears the result set
     */
    void clear()
    {
        dist_index_.clear();
        worst_dist_ = std::numeric_limits<DistanceType>::max();
        is_full_ = false;
    }

    /**
     *
     * @return Number of elements in the result set
     */
    size_t size() const
    {
        return dist_index_.size();
    }

    /**
     * Radius search result set always reports full
     * @return
     */
    bool full() const
    {
        return is_full_;
    }

    /**
     * Add another point to result set
     * @param dist distance to point
     * @param index index of point
     * Pre-conditions: capacity_>0
     */
    void addPoint(DistanceType dist, size_t index, int tree = 0)
    {
    	if (dist>=worst_dist_) return;

    	if (dist_index_.size()==capacity_) {
    		// if result set if filled to capacity, remove farthest element
    		std::pop_heap(dist_index_.begin(), dist_index_.end());
        	dist_index_.pop_back();
    	}

    	// add new element
    	dist_index_.push_back(DistIndex(dist,index));
    	if (is_full_) { // when is_full_==true, we have a heap
    		std::push_heap(dist_index_.begin(), dist_index_.end());
    	}

    	if (dist_index_.size()==capacity_) {
    		if (!is_full_) {
    			std::make_heap(dist_index_.begin(), dist_index_.end());
            	is_full_ = true;
    		}
    		// we replaced the farthest element, update worst distance
        	worst_dist_ = dist_index_[0].dist_;
        }
    }

    /**
     * Copy indices and distances to output buffers
     * @param indices
     * @param dists
     * @param num_elements Number of elements to copy
     * @param sorted Indicates if results should be sorted
     */
    void copy(size_t* indices, DistanceType* dists, size_t num_elements, bool sorted = true)
    {
    	if (sorted) {
    		// std::sort_heap(dist_index_.begin(), dist_index_.end());
    		// sort seems faster here, even though dist_index_ is a heap
    		std::sort(dist_index_.begin(), dist_index_.end());
    	}
    	else {
    		if (num_elements<size()) {
    			std::nth_element(dist_index_.begin(), dist_index_.begin()+num_elements, dist_index_.end());
    		}
    	}

    	size_t n = std::min(dist_index_.size(), num_elements);
    	for (size_t i=0; i<n; ++i) {
    		*indices++ = dist_index_[i].index_;
    		*dists++ = dist_index_[i].dist_;
    	}
    }

    DistanceType worstDist() const
    {
    	return worst_dist_;
    }

private:
    size_t capacity_;
    DistanceType worst_dist_;
    std::vector<DistIndex> dist_index_;
    bool is_full_;
};

}

#endif
