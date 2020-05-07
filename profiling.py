from builtins import object
from timeit import default_timer # import the best platform- and version-specific timer

"""
Performance timer for profiling;
call start() and stop() around measured code;
stats() will return cummulative time spent in section
"""
class Timer(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self._totalDuration = 0.0
        self._numIntervals = 0
        self._start = None

    def start(self):
        if( not self._start is None ):
            raise Exception("Timer started while already running (duration=%f secs)" % (default_timer()-self._start))
        self._start = default_timer()

    def abort(self):
        self._start = None

    def stop(self):
        duration = default_timer() - self._start
        assert(duration >= 0.0)
        self._start = None
        self._totalDuration += duration
        self._numIntervals += 1

    def stats(self):
        return (self._totalDuration, self._numIntervals)

