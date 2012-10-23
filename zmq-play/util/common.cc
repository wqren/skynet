#include "util/common.h"

namespace util {

using std::string;
using std::vector;

double Now() {
  timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);
  return tp.tv_sec + 1e-9 * tp.tv_nsec;
}

string Hostname() {
  char hostnameBuffer[1024];
  gethostname(hostnameBuffer, 1024);
  return hostnameBuffer;
}

timeval timevalFromDouble(double t) {
  timeval tv;
  tv.tv_sec = int(t);
  tv.tv_usec = (t - int(t)) * 1e6;
  return tv;
}

timespec timespecFromDouble(double t) {
  timespec tv;
  tv.tv_sec = int(t);
  tv.tv_nsec = (t - int(t)) * 1e9;
  return tv;
}

void Sleep(double time) {
  timespec req = timespecFromDouble(time);
  nanosleep(&req, NULL);
}


}  // namespace util
