#ifndef LEGION_COMMON_H_
#define LEGION_COMMON_H_

#include <boost/cstdint.hpp>
#include <boost/function.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <time.h>
#include <signal.h>

namespace util {

static inline uint64_t rdtsc() {
  uint32_t hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return (((uint64_t) hi) << 32) | ((uint64_t) lo);
}

static inline void breakpoint() {
  struct sigaction oldAct;
  struct sigaction newAct;
  newAct.sa_handler = SIG_IGN;
  sigaction(SIGTRAP, &newAct, &oldAct);
  raise(SIGTRAP);
  sigaction(SIGTRAP, &oldAct, NULL);
}

double Now();
std::string Hostname();
timeval timevalFromDouble(double t);
timespec timespecFromDouble(double t);

void Sleep(double sleepTime);

}

#endif /* LEGION_COMMON_H_ */
