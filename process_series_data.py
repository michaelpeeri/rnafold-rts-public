from __future__ import division
#
#
#
from builtins import range
from six import string_types
from builtins import object
import sys
import codecs
import json
from math import ceil
from datetime import datetime
from collections import Counter, Iterable
from bz2 import BZ2File
import argparse
import numpy as np
import config
import mysql_rnafold as db
from data_helpers import CDSHelper, countSpeciesCDS, getSpeciesName, getSpeciesFileName, SpeciesCDSSource, numItemsInQueue, getAllComputedSeqsForSpecies, splitLongSequenceIdentifier, decompressSeriesRecord, decodeJsonSeriesRecord, getAllNativeCDSsForSpecies, decompressNucleicSequence, getCrc
from rate_limit import RateLimit



rl = RateLimit(30)


"""
Generate all decoded series records (results) for the given species or list of species
"""
def readSeriesResultsForSpecies( seriesSourceNumber, species, minShuffledGroups=20, maxShuffledGroups=20, shuffleType=db.Sources.ShuffleCDSv2_python, cdsFilter=None, returnCDS=True ):
    if isinstance(species, Iterable):  # usually, species will be a sequence of numeric taxid values
        if isinstance(species, string_types):
            raise Exception("species cannot be string")
        # all set - proceed...
    else:
        species = (species,) # assume we got a single (numeric) taxid value
    assert(minShuffledGroups <= maxShuffledGroups)

    for taxIdForProcessing in species:
        print("Procesing %d sequences for tax-id %d (%s)..."
              % (countSpeciesCDS(taxIdForProcessing),
                 taxIdForProcessing,
                 getSpeciesName(taxIdForProcessing)))

        computed = getAllComputedSeqsForSpecies(seriesSourceNumber, taxIdForProcessing, maxShuffledGroups, shuffleType=shuffleType )
        computedIds = frozenset(list(computed.keys()))
        print("Collecting data from %d computation results..." % len(computed))


        skipped = 0
        selected = 0
        alreadyCompleted = 0

        # Iterate over all CDS entries for this species
        for protId in SpeciesCDSSource(taxIdForProcessing):
            cds = CDSHelper(taxIdForProcessing, protId)

            if (not cdsFilter is None) and (not cdsFilter(cds)):
                continue

            cdsSeqId = cds.seqId()

            shuffledIds = cds.shuffledSeqIds(shuffleType=shuffleType)

            # How many shuffles (for this cds) exist in the data we found?
            computedShufflesCount = len(computedIds.intersection(frozenset(shuffledIds)))

            if( computedShufflesCount < minShuffledGroups or (not cdsSeqId in computedIds) ):
                #print("%s - found only %d groups, skipping" % (protId, computedShufflesCount))
                skipped += 1
                continue

            # Get the computed results for this CDS
            seqIds = [cds.seqId()]
            seqIds.extend( cds.shuffledSeqIds(shuffleType=shuffleType) )
            if( len(seqIds) > maxShuffledGroups+1 ):
                seqIds = seqIds[:maxShuffledGroups+1]
            results = [ computed.get(x) for x in seqIds ]


            if( results is None or len([() for x in results if not x is None]) < minShuffledGroups ):
                print("Not enough results found for %s" % protId)
                skipped += 1
                continue

            # Decode the results
            results = list([decodeJsonSeriesRecord(decompressSeriesRecord(x)) if not x is None else None for x in results])
            
            if( returnCDS ):
                yield {"taxid":taxIdForProcessing, "content":results, "cds":cds}
            else:
                yield {"taxid":taxIdForProcessing, "content":results}
            del results
            del cds
            selected += 1

            if( rl()):
                print("# %s - %d records included, %d records skipped" % (datetime.now().isoformat(), selected, skipped))


"""
Read raw series results (i.e. values computed on the sequences)
"""
def readSeriesResultsForSpeciesWithSequence( seriesSourceNumber, species, minShuffledGroups=20, maxShuffledGroups=20, shuffleType=db.Sources.ShuffleCDSv2_python, cdsFilter=None, returnCDS=True ):
    if isinstance(species, Iterable):  # usually, species will be a sequence of numeric taxid values
        if isinstance(species, string_types):
            raise Exception("species cannot be string")
        # all set - proceed...
    else:
        species = (species,) # assume we got a single (numeric) taxid value

    for taxIdForProcessing in species:
        # Get seqs
        allseqs = getAllNativeCDSsForSpecies(taxIdForProcessing)
        
        for obj in readSeriesResultsForSpecies( seriesSourceNumber, (taxIdForProcessing,), minShuffledGroups=minShuffledGroups, maxShuffledGroups=maxShuffledGroups, shuffleType=shuffleType, cdsFilter=cdsFilter, returnCDS=returnCDS ):
            assert(obj["taxid"]==taxIdForProcessing)

            assert(len(obj["content"]) >= minShuffledGroups+1)

            cdsSeqId = obj["cds"].seqId()
            cdsSeq = decompressNucleicSequence(allseqs[cdsSeqId])
            obj["cds-seq"] = cdsSeq
            yield obj

        del allseqs

def convertResultsToMFEProfiles( results, maxShuffledGroups=20 ):
    for result in results:
        content = result["content"]  # 'content' contains an array of the json records (for each shuffle)
        profileLength = max( [len(x["MFE-profile"]) for x in content] ) # calculate the profile length (use the maximum, just in case not all profiles have the same length)

        # Verify the sequence CRC (optional) - makes sure the native CDS we got matches the one the profile was computed for
        # Note: This check is performed here to make sure the shuffles arrive in the expected order
        if "cds-seq" in result:
            recordedCRC = content[0]["seq-crc"]
            computedCRC = getCrc(result["cds-seq"])
            if (recordedCRC!=computedCRC):
                raise Exception("CRC mismatch detected in sequence %s" % result)

        profile = np.zeros((maxShuffledGroups+1, profileLength))
        for i, row in enumerate(content):
            profilei = row["MFE-profile"]
            if( len(profilei) < profileLength ):
                profilei[len(profilei):] = [None]*(profileLength-len(profilei)) # pad the list
            profile[i] = profilei

        result["profile-data"] = profile # Store the combined profile in the processed result
        yield result


def sampleProfilesFixedIntervals(results, startPosition=0, endPosition=5000, interval=10, reference="begin"):
    
    for result in results:
        fullProfile = result["profile-data"]
        #print(fullProfile)
        #print(fullProfile.shape)
        #positions = range(startPosition, min(endPosition, fullProfile.shape[1]), interval)
        if reference=="begin":
            maxPosition = min(endPosition, fullProfile.shape[1])
            result["profile-data"] = fullProfile[:, startPosition:maxPosition:interval]  # replace the full profile with the sampled profile
            
        elif reference=="end":
            lastPosition = fullProfile.shape[1]-1
            #print("lastPosition: {}".format(lastPosition))
            #firstPosition = (lastPosition - startPosition) % interval
            #print("firstPosition: {}".format(firstPosition))
            firstPosition = lastPosition - endPosition
            #print("firstPosition: {}".format(firstPosition))
            #print("startPosition: {}".format(startPosition))
            result["profile-data"] = fullProfile[:, firstPosition:(lastPosition+interval):interval]  # replace the full profile with the sampled profile
            #print(result["profile-data"].shape)
            
        elif reference=="stop3utr":
            stopCodonPos = result['content'][0]['stop-codon-pos']
            assert(stopCodonPos%3 == 0)
            assert(stopCodonPos>4)

            # Determine the profile reference point (in order to correctly align the data for different CDSs)
            lastFullCDSwindow = stopCodonPos+2 
            # Sample the existing data according to the profile definition
            values = fullProfile[:, max(0, lastFullCDSwindow-(endPosition//2)):lastFullCDSwindow+(endPosition//2):interval]

            
            #print("uut: {} -> {}".format( fullProfile.shape, values.shape ))
            #print("uut: stop:{} lastFullCDSWindow: {} [:, {}:{}:{}]".format( stopCodonPos, lastFullCDSwindow, max(0, lastFullCDSwindow-(endPosition//2)), lastFullCDSwindow+(endPosition//2), interval ))
            
            #values.shape = (21,real_len)
            if lastFullCDSwindow < endPosition//2:
                paddingLeft =  endPosition//2 - lastFullCDSwindow
                #print("uut: pad: endCodonPos {}: paddingLeft: {} endPos: {} endPos//2: {}".format(stopCodonPos, paddingLeft, endPosition, endPosition//2))
                values2 = np.hstack( (np.full( (values.shape[0], paddingLeft), np.nan ), values ) )
                values = values2
                #print("uut: pad: {}".format(values.shape))

            result["profile-data"] = values

            # print("============"*5)
            # print(values.shape)
            # print("---nan---")     # 39 empties
            # print(np.sum( np.isnan(values), axis=0))
            # print(np.sum( np.isnan(values), axis=1))
            # print("---0.0---")   # end missing values
            # print(np.sum( values==0, axis=0))
            # print(np.sum( values==0, axis=1))

            
            
        else:
            assert(False)
            
        del fullProfile
        yield result


def profileLength(profileSpec):
    if profileSpec[2] == "begin":
        return int(ceil(float(profileSpec[0]-profileSpec[3]) / profileSpec[1]))  # should be equal to len(range(0, spec[0], spec[1]))
    elif profileSpec[2] == "end":
        return (profileSpec[0] // profileSpec[1])+1
    elif profileSpec[2] == "stop3utr":
        return (profileSpec[0] // profileSpec[1])
    else:
        assert(False)

def profileElements(profileSpec):
    if profileSpec[2] == "begin":
        return list(range(profileSpec[3], profileSpec[0], profileSpec[1]))
    elif profileSpec[2] == "end":
        return list(range(-profileSpec[0], 1, profileSpec[1]))
    elif profileSpec[2] == "stop3utr":
        return list(range(-(profileSpec[0]//2), (profileSpec[0]//2), profileSpec[1] ))
    else:
        assert(False)

"""
Find the position (index) of the "edge" (0nt) element
"""
def profileEdgeIndex(profileSpec):
    raise Exception("Not impl")

    # if profileSpec[2] == "begin":
    #     return list(range(profileSpec[3], profileSpec[0], profileSpec[1])).index(0)
    # elif profileSpec[2] == "end":
    #     return list(range(-profileSpec[0], 1, profileSpec[1])).index(0)
    # else:
    #     assert(False)

    
class MeanProfile(object):
    def __init__(self, length):
        self._a = np.zeros((length,))
        self._n = np.zeros((length,), dtype=np.int)

    def add(self, vals):
        assert(vals.ndim==2)
        affectedSpan = vals.shape[1]  # 'vals' may be narrower than the profile. This operation only affects the overlapping section.
        contrib = vals.copy()
        contrib[np.isnan(vals)] = 0.0   # Convert nan's to 0 when summing
        self._a[:affectedSpan] += contrib.sum(axis=0)

        newcounts = vals.shape[0] - np.isnan(vals).sum(axis=0)  # Update the number of counts (excluding nan's) in each column
        self._n[:affectedSpan] += newcounts

    """
    Compute the mean value for each column.
    If no values are included for a column, its value is np.nan.
    """
    def value(self):
        out = np.full(self._a.shape, np.nan, dtype=float)

        ind = self._n>0  # mean is only defined for n>0 items
        out[ind] = self._a[ind] / self._n[ind]
        return out

    """
    Return the number of values included for each column
    """
    def counts(self):
        return self._n


   
    

def testMeanProfile():
    # Test rational: calculate mean profiles, for normal values centered around a specified mean value X.
    # Some rows and columns will be missing.
    # However, if the mean values are calculated correctly, the means of all remaining values should
    # be centered around X.
    from random import randint
    L  = 20     # Profile width
    N1 = 100    # Number of value matrices to process
    N2 = 10000  # Number of vectors in each matrix
    
    m = MeanProfile(L)  # 'm' will be our test profile

    for i in range(N1):
        newvals = np.random.normal(loc=0.456, size=(N2,L))
        if( randint(1,10) >= 6 ): # Sometimes erase a full row (but only from the first 10 rows)
            newvals[randint(1,5),:] = np.full((L,), np.nan)

        if( randint(1,10) >= 6 ): # Sometimes erase a full column (but only from the first 10 columns)
            newvals[:,randint(1,5)] = np.full((N2,), np.nan)
            
        m.add(newvals)

    results = m.value()
    #print(results)
    counts = m.counts()
    #print(counts)

    assert(np.all(np.abs( results - 0.456 ) < 5e-3))
    assert(np.all( counts > 0.75*N1*N2 ) )




"""
Calculte GC content for a sequence, with support for down-sampling.
If stepSize>1, the result will be down-sampled by a factor equal to stepSize.
For example:
calcSampledGCcontent("aattacca", 1)   --> [  0   0   0   0   0   1   1   0]
calcSampledGCcontent("aattacca", 2)   --> [0.0 0.0 0.5 0.5]

Note: The boundary should calculated correctly (as if the sequence was padded with 'n's to complete the last window)
Note: In the rare case when a window does not contain any valid nucleotides (e.g., only 'n's), the result for that window is nan (0/0).
"""
def calcSampledGCcontent(seq, stepSize=10):
    assert(stepSize>0)
    seq = seq.lower()
    if( len(seq) % stepSize > 0):
        padding = stepSize - (len(seq) % stepSize)
        seq = seq + 'n'*padding
    assert(len(seq) % stepSize == 0)

    def downsampleVector(vec, n):
        assert(vec.size%n == 0)
        vec = np.reshape(vec, (-1,n))
        return np.sum(vec, axis=1)
    
        
    GCcounts = np.array([x=='c' or x=='g' for x in seq], dtype=float)
    Totalcounts = np.array([x=='a' or x=='c' or x=='t' or x=='g' for x in seq], dtype=float)

    gc = downsampleVector(GCcounts, stepSize)
    total = downsampleVector(Totalcounts, stepSize)

    return gc/total



def testSampledGC():
    for i in range(30):
        v = calcSampledGCcontent('g'*i)
        print("%d -> %s" % (i, v))
        assert(all(v==1.0))
    for i in range(30):
        v = calcSampledGCcontent('a'*i)
        print("%d -> %s" % (i, v))
        assert(all(v==0.0))
    for i in range(29):
        v = calcSampledGCcontent('c'*i + 'n')
        print("%d -> %s" % (i+1, v))
        assert(np.all(np.logical_or(np.isclose(v,1.0), np.isnan(v))))
    for i in range(15):
        v = calcSampledGCcontent('tc'*i + 'n')
        print("%d -> %s" % (i+1, v))
        assert(np.all(np.logical_or(np.isclose(v,0.5), np.isnan(v))))
    for i in range(15):
        v = calcSampledGCcontent('cgcga'*i + 'nnn')
        print("%d -> %s" % (i+1, v))
        assert(np.all(np.logical_or(np.isclose(v,0.80), np.isnan(v))))

        
def runAllTests(repeats=10):
    print("Running unit tests...")
    for n in range(repeats):
        print("Repeat %d out of %d..." % (n+1, repeats))
        testMeanProfile()
        testSampledGC()
        
    return 0

if __name__=="__main__":
    from sys import exit
    exit(runAllTests())
    
    
