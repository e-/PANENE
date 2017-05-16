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

  Schedule(int addNewPointOps_ = -1, int updateIndexOps_ = -1, int updateTableOps_ = -1) : addNewPointOps(addNewPointOps_), updateIndexOps(updateIndexOps_), updateTableOps(updateTableOps_)
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
