from builtins import object
import time

class RateLimit(object):
    def __init__(self, limitSecs = 60):
        self._limit = limitSecs
        self._last = time.time() - (self._limit+1)

    def __call__(self):
        now = time.time()
        if( (now - self._last) > self._limit ):
            self._last = now
            return True
        else:
            return False
