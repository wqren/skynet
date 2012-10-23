#include "zmq.hpp"
#include <cstdio>
#include "util/logging.h"

int main(int argc, char** argv) {
  zmq::context_t ctx(4);
  zmq::socket_t client(ctx, ZMQ_PUSH);
  client.connect("tcp://localhost:9999");

  zmq::message_t rep;
  zmq::message_t req(6);
  memcpy((void*)req.data(), "Hello", 5);

  int count = strtol(argv[1], NULL, 10);
  for (int i = 0; i < count; ++i) {
    client.send(req);
  }

//  for (int i = 0; i < count; ++i) {
//    client.recv(&rep);
//  }

  printf("Done.");
}
