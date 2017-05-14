#ifndef panene_schedule_h
#define panene_schedule_h

#include <iostream>

namespace panene
{

class Schedule
{
public:
  int addNewPointOps;
  int updateIndexOps;
  int updateTableOps;

  Schedule() : addNewPointOps(-1), updateIndexOps(-1), updateTableOps(-1) {}

  Schedule(int addNewPointOps_, int updateIndexOps_, int updateTableOps_) : addNewPointOps(addNewPointOps_), updateIndexOps(updateIndexOps_), updateTableOps(updateTableOps_)
  {}

  friend std::ostream& operator<<( std::ostream& os, const Schedule& schedule) {
    os << "Schedule(addNewPointOps: " << schedule.addNewPointOps << ", " 
       << "updateIndexOps: " << schedule.updateIndexOps << ", " 
       << "updateTableOps: " << schedule.updateTableOps << ")";
    return os;  
  }
};

}

#endif
