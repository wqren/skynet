#include <boost/mpi.hpp>
#include <string>
#include <boost/unordered_map.hpp>

using namespace boost;

static unordered_map<int32_t, int32_t> theMap;

struct Request {
  int32_t key;
  template <class Arc>
  void serialize(Arc& ar, const unsigned int) { ar & key; }
};

struct Response {
  int32_t key;
  int32_t value;
  template <class Arc>
  void serialize(Arc& ar, const unsigned int) { ar & key & value; }
};

enum RequestTypes {
  kGetRequest = 1,
  kGetResponse = 2
};

#define SERIALIZABLE(R)\
  BOOST_IS_MPI_DATATYPE(R)\
  BOOST_IS_BITWISE_SERIALIZABLE(R)

SERIALIZABLE(Request) 
SERIALIZABLE(Response)

static const int kNumServers = 4;
static const int kNumRequests = 1000 * 1000 * 50;
static const int kHashTableSize = 50;

int main(int argc, char* argv[]) 
{
  mpi::environment env(argc, argv);
  mpi::communicator world;

  Request req;
  Response resp;

  int numClients = world.size() - kNumServers;

  if (world.rank() < kNumServers) {
    for (int i = 0; i < kHashTableSize; ++i) {
      theMap[i] = i;
    }

    printf("Server ready...\n");

    for (int i = 0; i < kNumRequests * numClients / kNumServers; ++i) {
      mpi::status st = world.recv(mpi::any_source, kGetRequest, req);
      //printf("Got msg...\n");
      world.send(st.source(), kGetResponse, resp);
    }
  } else {
    printf("Started client.\n");
    for (int i = 0; i < kNumRequests; ++i) {
      world.send(i % kNumServers, kGetRequest, req);
      world.recv(i % kNumServers, kGetResponse, resp);
      //printf("Got response...\n");
    }
    printf("Done...\n");
  }

  return 0;
}
