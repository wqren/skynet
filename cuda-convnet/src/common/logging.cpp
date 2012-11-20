#include <libgen.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include "common/logging.h"

using namespace std;

LogLevel currentLogLevel = kLogInfo;
static const char* logLevels[5] = { "D", "I", "W", "E", "F" };

double Now() {
  timespec tp;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tp);
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

double get_processor_frequency() {
  double freq;
  int a, b;
  FILE* procinfo = fopen("/proc/cpuinfo", "r");
  while (fscanf(procinfo, "cpu MHz : %d.%d", &a, &b) != 2) {
    fgetc(procinfo);
  }

  freq = a * 1e6 + b * 1e-4;
  fclose(procinfo);
  return freq;
}

void logAtLevel(LogLevel level, const char* path, int line, const char* fmt, ...) {
  if (level < currentLogLevel) {
    return;
  }

  char buffer[4096];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer, 4095, fmt, args);
  va_end(args);

  char file[256];
  memcpy(file, path, 255);
  basename(file);

  double subSecond = Now();

  fprintf(stderr, "%s %4.3f [%5d] %s:%3d %s\n", logLevels[level], subSecond,
      getpid(), file, line, buffer);

  fflush(stderr);
  if (level == kLogFatal) {
    abort();
  }
}
