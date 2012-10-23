#include <boost/program_options.hpp>
#include <boost/unordered_map.hpp>
#include <mpi/mpi.h>
#include <string>
#include <vector>

#include "util/common.h"
#include "util/logging.h"

namespace opt = boost::program_options;
using std::string;
using std::vector;

int main(int argc, char* argv[]) {
  MPI::Init();
  MPI::Intracomm world = MPI::COMM_WORLD;

  long epochs, dataSize;
  opt::options_description desc;
  desc.add_options()
      ("epochs", opt::value<long>(&epochs)->default_value(10000), "# req")
      ("size", opt::value<long>(&dataSize)->default_value(100000), "req size");

  opt::variables_map opt_map;
  opt::store(opt::parse_command_line(argc, argv, desc), opt_map);
  opt::notify(opt_map);

  int numPeers = world.Get_size();
  int myid = world.Get_rank();
  dataSize /= numPeers;
  vector<string> resp;
  resp.resize(numPeers);
  for (int i = 0; i < numPeers; ++i) {
    resp[i].resize(dataSize);
  }

  std::string req;
  req.resize(dataSize);

  std::vector<double> times;
  for (int i = 0; i < epochs; ++i) {
    double start = util::Now();
    std::vector<MPI::Request> sends;
    std::vector<MPI::Request> recvs;
    for (int j = 0; j < numPeers; ++j) {
      if (myid == j)
        continue;
      sends.push_back(
          world.Isend(req.data(), req.size(), MPI::UNSIGNED_CHAR, j, 0));
    }

    for (int j = 0; j < numPeers; ++j) {
      if (myid == j)
        continue;
      recvs.push_back(
          world.Irecv(&resp[j][0], dataSize, MPI::UNSIGNED_CHAR, j, 0));
    }
    for (int j = 0; j < sends.size(); ++j) {
      sends[j].Wait();
    }

    for (int j = 0; j < recvs.size(); ++j) {
      recvs[j].Wait();
    }

    times.push_back(util::Now() - start);
  }

  printf("%s", util::ToString(times).c_str());
  MPI::Finalize();
  return 0;
}
