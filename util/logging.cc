#include "util/logging.h"

#include <libgen.h>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <time.h>

#include "util/common.h"
#include "util/logging.h"

using std::string;

namespace util {

LogLevel currentLogLevel = kLogInfo;
static const char* logLevels[5] = { "D", "I", "W", "E", "F" };

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
  strncpy(file, path, 255);
  basename(file);

  char timeBuffer[4096];
  double subSecond = Now();
  // subSecond -= int(subSecond);
  // sprintf("%0.3f", subSecond);

  time_t now = time(NULL);
  struct tm now_t;
  localtime_r(&now, &now_t);
  strftime(timeBuffer, 4096, "%Y%m%d:%H%M%S", &now_t);

  fprintf(stderr, "%s %.3f [%5d] %s:%3d %s\n", logLevels[level], subSecond,
      getpid(), file, line, buffer);

  fflush(stderr);
  if (level == kLogFatal) {
    abort();
  }
}

} // namespace util
