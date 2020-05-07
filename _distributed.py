from __future__ import division
from builtins import str
from builtins import range
from builtins import object
import json
import re
from hashlib import md5
from struct import unpack
from dask.distributed import Client, wait
from distributed.diagnostics import progress

schedulerConfig = None
with open('dask-scheduler.json', 'r') as f:
    schedulerConfig = json.loads(f.read())

assert(schedulerConfig['type']=='Scheduler')
schedulerAddress = re.match('tcp://(\d+.\d+.\d+.\d+(:\d+)?)', schedulerConfig['address']).group(1)

print("Dask scheduler address: %s" % schedulerAddress)

def open():
    scheduler = Client(schedulerAddress)
    #scheduler = Client('127.0.0.1:8786')
    print("Running on %d cores" % sum(scheduler.ncores().values()))
    return scheduler



"""
Hash for uniformly mapping long unique string keys to integers in the range 0..N-1  (for the specified N = numGroups).
This can be used to evenly map work among N workers.
From preliminary testing, the uniformity seems to be within -+5% (but this depends on N and the total number of keys; when either is small (in the range of a few thousands), deviation can be much larger; test with your values to be sure).
"""
class UniformShardKey(object):
    def __init__(self, numGroups):
        self._N = numGroups

    def getKey(self, longUniqueObjectId):
        # Decode the 2nd 8-byte half of the md5 digest as an integer, and map to group using modulus
        return unpack('8xQ', md5(longUniqueObjectId).digest())[0] % self._N

def testUniformShardKey(N=1000):
    from collections import Counter
    
    hN = UniformShardKey(N)

    totalWork = 7000
    meanWorkPerGroup = float(totalWork)/N
    
    counts = Counter( [hN.getKey(str(x)) for x in range(70000000, 70000000+totalWork)] )
    workloadOfBusiestGroup = counts.most_common(1)[0][1]
    print("Parameters: total work: %.4g tasks; groups: %d" % (totalWork, N))
    print("Even workload: %.4g tasks/group" % meanWorkPerGroup)
    print("---------------------------------------------------------")
    print("Test results:")
    print("Busiest group: %.4g tasks" % workloadOfBusiestGroup)
    print("Deviation: %.2g%%" % (((workloadOfBusiestGroup/meanWorkPerGroup)-1.0)*100))

    
if __name__=="__main__":
    import sys
    if( len(sys.argv) <= 1 ):
        sys.exit(0)

    if sys.argv[1]=="restart":
        # Restart all dask workers, to force reloading of module code
        s = open()
        s.restart()
    elif sys.argv[1]=="testShardKey":
        # Test uniformity of the UniformShardKey()
        testUniformShardKey(100)
    else:
        raise Exception("Unknown command '%s'" % sys.argv[1])

    sys.exit(0)
