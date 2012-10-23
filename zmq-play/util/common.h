#ifndef LEGION_COMMON_H_
#define LEGION_COMMON_H_

#include <boost/function.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <time.h>


namespace util {

static inline uint64_t rdtsc() {
  uint32_t hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return (((uint64_t) hi) << 32) | ((uint64_t) lo);
}

double Now();
std::string Hostname();
timeval timevalFromDouble(double t);
timespec timespecFromDouble(double t);

void Sleep(double sleepTime);

}

#endif /* LEGION_COMMON_H_ */
