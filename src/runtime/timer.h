#include "bind.h"

using namespace kernelgen::bind::cuda;

#include <time.h>

namespace kernelgen { namespace runtime {

class timer
{
        timespec ts_start, ts_stop;
        cudaEvent_t ce_start, ce_stop;
        bool cuda, started;

public :

        static timespec get_resolution();

        timer(bool start = true);

        timespec start();
        timespec stop();

        double get_elapsed(timespec* start = NULL);
};

} }

