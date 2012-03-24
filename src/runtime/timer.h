#include <time.h>

namespace kernelgen { namespace runtime {

class timer
{
        timespec time_start, time_stop;
        bool started;

public :

        static timespec get_resolution();

        timer(bool start = true);

        timespec start();
        timespec stop();

        double get_elapsed(timespec* start = NULL);
};

} }

