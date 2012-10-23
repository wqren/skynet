#include <cstdio>
#include "util/logging.h"
#include "zmq.hpp"

int main(int argc, char** argv) {
  zmq::context_t ctx(4);
  zmq::socket_t responder(ctx, ZMQ_PULL);
  responder.bind("tcp://ib0:9999");

  while (true) {
    zmq::message_t req;
    responder.recv(&req, 0);

    EVERY_N((1000 * 1000), Log_Info("Working... %d.  Received size: %d", COUNT, req.size()));

    //zmq::message_t reply((void*)"World", 5, NULL);
    //responder.send(reply);
  }
  printf("...\n");
}
