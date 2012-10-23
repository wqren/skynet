#include "zmq.hpp"
#include <cstdio>
#include <boost/program_options.hpp>
#include "util/logging.h"

namespace opt = boost::program_options;

int main(int argc, char** argv) {
  long count, size;
  opt::options_description desc;
  desc.add_options()
      ("count", opt::value<long>(&count)->default_value(10000000), "# req")
      ("size", opt::value<long>(&size)->default_value(10), "req size");

  opt::variables_map opt_map;
  opt::store(opt::parse_command_line(argc, argv, desc), opt_map);
  opt::notify(opt_map);

  zmq::context_t ctx(4);
  zmq::socket_t client(ctx, ZMQ_PUSH);
  client.connect("tcp://192.168.5.25:9999");

  Log_Info("Requests: %.0fm, size: %d", count / 1e6, size);
  zmq::message_t rep;
  for (int i = 0; i < count; ++i) {
    zmq::message_t req(size);
    client.send(req);
  }

//  for (int i = 0; i < count; ++i) {
//    client.recv(&rep);
//  }

  printf("Done.");
}
