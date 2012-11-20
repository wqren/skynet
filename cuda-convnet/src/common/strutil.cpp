#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "common/strutil.h"

using std::vector;

StringPiece::StringPiece() :
    data_(NULL), len_(0) {
}
StringPiece::StringPiece(const StringPiece& s) :
    data_(s.data_), len_(s.size()) {
}
StringPiece::StringPiece(const string& s) :
    data_(s.data()), len_(s.size()) {
}
StringPiece::StringPiece(const string& s, int len) :
    data_(s.data()), len_(len) {
}
StringPiece::StringPiece(const char* c) :
    data_(c), len_(strlen(c)) {
}
StringPiece::StringPiece(const char* c, int len) :
    data_(c), len_(len) {
}

uint64_t StringPiece::hash() const {
  uint64_t out[2];
  MurmurHash3(data_, len_, 0, out);
  return out[0];
}

uint64_t StringPiece::hash(const StringPiece& sp) {
  return sp.hash();
}

string StringPiece::AsString() const {
  return string(data_, len_);
}

void StringPiece::strip() {
  while (len_ > 0 && isspace(data_[0])) {
    ++data_;
    --len_;
  }
  while (len_ > 0 && isspace(data_[len_ - 1])) {
    --len_;
  }
}
vector<StringPiece> StringPiece::split(StringPiece sp, StringPiece delim) {
  vector<StringPiece> out;
  const char* c = sp.data_;
  while (c < sp.data_ + sp.len_) {
    const char* next = c;

    bool found = false;

    while (next < sp.data_ + sp.len_) {
      for (int i = 0; i < delim.len_; ++i) {
        if (*next == delim.data_[i]) {
          found = true;
        }
      }
      if (found)
        break;

      ++next;
    }

    if (found || c < sp.data_ + sp.len_) {
      StringPiece part(c, next - c);
      out.push_back(part);
    }

    c = next + 1;
  }

  return out;
}

bool operator==(const StringPiece& a, const StringPiece& b) {
  return a.size() == b.size() && memcmp(a.data(), b.data(), a.size()) == 0;
}

bool operator==(const StringPiece& a, const char* b) {
  bool result = a == StringPiece(b);
  return result;
}

string StringPrintf(StringPiece fmt, ...) {
  va_list l;
  va_start(l, fmt);
  string result = VStringPrintf(fmt, l);
  va_end(l);

  return result;
}

string VStringPrintf(StringPiece fmt, va_list l) {
  char buffer[32768];
  vsnprintf(buffer, 32768, fmt.AsString().c_str(), l);
  return string(buffer);
}

string ToString(int32_t v) {
  return StringPrintf("%d", v);
}

string ToString(int64_t v) {
  return StringPrintf("%ld", v);
}

string ToString(double v) {
  return StringPrintf("%f", v);
}

string ToString(string v) {
  return v;
}
string ToString(StringPiece v) {
  return v.str();
}

const char* strnstr(const char* haystack, const char* needle, int len) {
  int nlen = strlen(needle);
  for (int i = 0; i < len - nlen; ++i) {
    if (strncmp(haystack + i, needle, nlen) == 0) {
      return haystack + i;
    }
  }
  return NULL;
}

inline uint64_t fmix(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdLLU;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53LLU;
  k ^= k >> 33;

  return k;
}

inline uint64_t rotl64(uint64_t x, int8_t r) {
  return (x << r) | (x >> (64 - r));
}

inline uint64_t getblock(const uint64_t * p, int i) {
  return p[i];
}

void MurmurHash3(const void * key, const int len, const uint32_t seed,
    void * out) {
  const uint8_t * data = (const uint8_t*) key;
  const int nblocks = len / 16;

  uint64_t h1 = seed;
  uint64_t h2 = seed;

  const uint64_t c1 = 0x87c37b91114253d5llu;
  const uint64_t c2 = 0x4cf5ad432745937fllu;

  //----------
  // body

  const uint64_t * blocks = (const uint64_t *) (data);

  for (int i = 0; i < nblocks; i++) {
    uint64_t k1 = getblock(blocks, i * 2 + 0);
    uint64_t k2 = getblock(blocks, i * 2 + 1);

    k1 *= c1;
    k1 = rotl64(k1, 31);
    k1 *= c2;
    h1 ^= k1;

    h1 = rotl64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = rotl64(k2, 33);
    k2 *= c1;
    h2 ^= k2;

    h2 = rotl64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;
  }

  //----------
  // tail

  const uint8_t * tail = (const uint8_t*) (data + nblocks * 16);

  uint64_t k1 = 0;
  uint64_t k2 = 0;

  switch (len & 15) {
  case 15:
    k2 ^= uint64_t(tail[14]) << 48;
  case 14:
    k2 ^= uint64_t(tail[13]) << 40;
  case 13:
    k2 ^= uint64_t(tail[12]) << 32;
  case 12:
    k2 ^= uint64_t(tail[11]) << 24;
  case 11:
    k2 ^= uint64_t(tail[10]) << 16;
  case 10:
    k2 ^= uint64_t(tail[9]) << 8;
  case 9:
    k2 ^= uint64_t(tail[8]) << 0;
    k2 *= c2;
    k2 = rotl64(k2, 33);
    k2 *= c1;
    h2 ^= k2;

  case 8:
    k1 ^= uint64_t(tail[7]) << 56;
  case 7:
    k1 ^= uint64_t(tail[6]) << 48;
  case 6:
    k1 ^= uint64_t(tail[5]) << 40;
  case 5:
    k1 ^= uint64_t(tail[4]) << 32;
  case 4:
    k1 ^= uint64_t(tail[3]) << 24;
  case 3:
    k1 ^= uint64_t(tail[2]) << 16;
  case 2:
    k1 ^= uint64_t(tail[1]) << 8;
  case 1:
    k1 ^= uint64_t(tail[0]) << 0;
    k1 *= c1;
    k1 = rotl64(k1, 31);
    k1 *= c2;
    h1 ^= k1;
  };

  //----------
  // finalization

  h1 ^= len;
  h2 ^= len;

  h1 += h2;
  h2 += h1;

  h1 = fmix(h1);
  h2 = fmix(h2);

  h1 += h2;
  h2 += h1;

  ((uint64_t*) out)[0] = h1;
  ((uint64_t*) out)[1] = h2;
}
