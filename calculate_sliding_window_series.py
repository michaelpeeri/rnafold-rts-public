# Create mRNA folding energy profiles for sequences, by calculation energy for windows missing from the profile
# (Read sequences and existing results, do the processing for missing values, and store the result in the sequence's entry)
# This module can be used as a stand-alone worker process for calculating RNA folding energy profiles, or expose this functionality for use elsewhere
from builtins import str
from builtins import map
from builtins import zip
from builtins import range
from builtins import object
import os
import subprocess
import gzip
import json
import logging
import profiling
from numpy import nan, isnan
from Bio.Seq import Seq
from itertools import compress
from time import sleep
from datetime import datetime, timedelta
import argparse
import mysql_rnafold as db
from data_helpers import CDSHelper, RegionsOfInterset, countSpeciesCDS, getCrc, QueueSource, createWorkerKey, updateJobStatus_ItemCompleted, getSpeciesTemperatureInfo, getSpeciesTranslationTable
from runningstats import RunningStats
from rate_limit import RateLimit
from rnafold_vienna import RNAfold_direct
import config


# hold configuration arguments (for either stand-alone or module use)
args = None

class DummyTimer(object):
    def start(self): pass
    def stop(self): pass
    def stats(self): return [None]

useProfiling = False
if useProfiling:
    timerForPreFolding  = profiling.Timer()
    timerForFolding     = profiling.Timer()
    timerForPostFolding = profiling.Timer()
else:
    timerForPreFolding  = DummyTimer()
    timerForFolding     = DummyTimer()
    timerForPostFolding = DummyTimer()
    

def parseOption(possibleValues, name):
    def checkOption(value):
        if value in possibleValues:
            return value
        else:
            raise argparse.ArgumentTypeError("Unknown %s '%s', allowed values: %s" % (name, value, ",".join(possibleValues)))
    return checkOption


translateAllDeterminedNucleotides = str.maketrans("acgtACGT", "&&&&&&&&")
translateGCNucleotides            = str.maketrans("gcGC",     "&&&&")
translatePurineNucleotides        = str.maketrans("agAG",     "&&&&")
def calcWindowGCContent(seq:str) -> float:
    allCount = seq.translate( translateAllDeterminedNucleotides ).count('&')
    if allCount==0:
        return nan
    
    gcCount  = seq.translate( translateGCNucleotides            ).count('&')

    return gcCount/allCount

def calcWindowPurineContent(seq:str) -> float:
    allCount     = seq.translate( translateAllDeterminedNucleotides ).count('&')
    if allCount==0:
        return nan
    
    purineCount  = seq.translate( translatePurineNucleotides        ).count('&')

    return purineCount/allCount

def calcWindowStopCodonContent(seq:str, translationTable:int =1, phase:int =0) -> float:
    codonLength = (len(seq)-(3-phase))//3
    start = 3-phase
    end = start + codonLength*3
    assert(end <= len(seq))
    codonSeq = seq[start:end]
    assert(len(codonSeq)%3==0)
    translatedSeq = Seq( codonSeq ).translate( table=translationTable )
    
    numStopCodons     = str(translatedSeq).count('*')

    return numStopCodons/codonLength


"""
Check if a timer has expired (used to implement running-time limit)
"""
class RunUntil(object):
    def __init__(self, duration=timedelta(0, 0, 0, 0, 5) ):
        self._endTime = datetime.now() + duration
        print("Program will exit at %s" % self._endTime)

    def isExpired(self):
        return datetime.now() > self._endTime


def round4(x):
    if x is None:
        return None
    else:
        return round(x,4)

test = [-4, None, None, None, None, None, None, None, None, None, -14, None, None, None, None, None, None, None, None, None, 26, None, None, None, None, None, None, None, None, None, 16, None, None, None, None, None, None, None, None, None, 6, None, None, None, None, None, None, None, None, None, -4, None,  None, None, None, None, None, None, None, None, -14, None, None, None, None, None, None, None, None, None, 26, None, None, None, None, None, None, None, None, None, 16, None, None, None, None, None, None, None, None, None, 6, None, None, None, None, None, None, None, None, None, -4, None, None, None, None, None, None, None, None, None, -14, None, None, None, None, None, None, None, None, None, 26, None, None, None, None, None, None, None, None, None, 16, None, None, None, None, None, None, None, None, None, 6, None, None, None, None, None, None, None, None, None, -4, None, None, None, None, None, None, None, None, None, -14, None, None, None, None, None, None, None, None, None, 26, None, None, None, None, None, None, None, None, None, 16, None, None, None, None, None, None, None, None, None, 6, None, None, None, None, None, None, None, None, None, -4, None, None, None, None, None, None, None, None, None, -14, None, None, None, None, None, None, None, None, None, 26, None, None, None, None, None, None, None, None, None, 16, None, None, None, None, None, None, None, None, None, 6, None, None, None, None, None, None, None, None, None, -4, None, None, None, None, None, None, None, None, None, -14, None, None, None, None, None, None, None, None, None, 26, None, None, None, None, None, None, None, None, None, 16, None, None, None, None, None, None, None, None, None, 6, None, None, None, None, None, None, None, None, None, -4, None, None, None, None, None, None, None, None, None, -14, None, None, None, None, None, None, None, None, None, 26, None, None, None, None, None, None, None, None, None, 16, None, None, None, None, None, None, None, None, None, 6, None, None, None, None, None, None, None, None, None, -4, None, None, None, None, None, None, None, None, None, -14, None, None, None, None, None, None, None, None, None, 26, None, None, None, None, None, None, None, None, None, 16, None, None, None, None, None, None, None, None, None, 6, None, None, None, None, None, None, None, None, None, -4, None, None, None, None, None, None, None, None, None, -14]
def prettyPrintProfile(profile):
    for start in range(0, len(profile), 10):
        elements = profile[start:start+10]
        print("[{}]\t{}\t[{}]".format(start, ",\t".join(["{}".format(x) for x in elements]), start+len(elements)-1))
        

class CalculateSlidingWindowRandomizedComparisonSeries(object):
    def __init__(self, windowWidth, logfile=None, debugDoneWriteResults=False, computationTag="rna-fold-window-40-0", seriesSourceNumber=db.Sources.RNAfoldEnergy_SlidingWindow40_v2):
        if logfile is None:
            self._logfile = open('/dev/null', 'w')
        else:
            self._logfile = logfile

        self._debugDoneWriteResults = debugDoneWriteResults

        self._compuatationTag = computationTag
        self._seriesSourceNumber = seriesSourceNumber
        self._windowWidth = windowWidth


            
    def calculateMissingWindowsForSequence(self, taxId, protId, seqIds, requestedShuffleIds, firstWindow, lastWindowStart, windowStep, reference="begin", shuffleType=db.Sources.ShuffleCDSv2_python, debug=False):

        timerForPreFolding.start()
        logging.warning("Parameters: %d %s %s %s %d %d %s %d" % (taxId, protId, seqIds, requestedShuffleIds, lastWindowStart, windowStep, reference, shuffleType))
        f = self._logfile

        assert(len(seqIds)>0)
        assert(len(seqIds)==len(requestedShuffleIds))

        # ------------------------------------------------------------------------
        # Obtain species-dependent properties needed for some calculations
        # ----------------
        # Optimal Temp
        optimalSpeciesGrowthTemperature = None
        if( self._seriesSourceNumber == db.Sources.RNAfoldEnergy_SlidingWindow40_v2_native_temp ):
            (numericalProp, _) = getSpeciesTemperatureInfo(taxId)
            optimalSpeciesGrowthTemperature = numericalProp[0]

            if optimalSpeciesGrowthTemperature is None:
                raise Exception("No temperature value for taxid={}, can't calculate native-temperature folding profile...".format(taxId))
            else:
                optimalSpeciesGrowthTemperature = float(optimalSpeciesGrowthTemperature)
                assert(optimalSpeciesGrowthTemperature >= -30.0 and optimalSpeciesGrowthTemperature <= 150.0)
        # ----------------
        # Genomic translation table
        genomicTranslationTable = None
        if( self._seriesSourceNumber in (db.Sources.StopCodon_content_SlidingWindow30, db.Sources.StopCodon_content_SlidingWindow40, db.Sources.StopCodon_content_SlidingWindow50 )):
            genomicTranslationTable = getSpeciesTranslationTable(taxId)
            assert(genomicTranslationTable>0 and genomicTranslationTable<=31)
            

        if( reference != "begin" and reference != "end" and reference != "stop3utr"):
            timerForPreFolding.stop()
            e = "Specificed profile reference '%s' is not supported!" % reference
            logging.error(e)
            raise Exception(e)

        # We will process all listed shuffle-ids for the following protein record
        if( reference == "begin" or reference == "end" ):
            regionOfInterest = RegionsOfInterset.CDSonly
        elif reference == "stop3utr":
            regionOfInterest = RegionsOfInterset.CDSand3UTR
        else:
            assert(False)
            
        cds = CDSHelper( taxId, protId, regionOfInterest=regionOfInterest )

        if( cds.length() < self._windowWidth ):
            e = "Refusing to process item %s because the sequence length (%d nt) is less than the window size (%d nt)\n" % (itemToProcess, cds.length(), self._windowWidth)
            f.write(e)
            logging.error(e)
            timerForPreFolding.stop()
            raise Exception(e)

        # Create a list of the windows we need to calculate for this CDS
        if reference == "begin":
            requestedWindowStarts = frozenset(list(range(0, min(lastWindowStart+1, cds.length()-self._windowWidth-1), windowStep)))
            if( len(requestedWindowStarts) == 0):
                e = "No windows exist for calculation taxid=%d, protId=%s, CDS-length=%d, lastWindowStart=%d, windowStep=%d, windowWidth=%d - Skipping...\n" % (taxId, protId, cds.length(), lastWindowStart, windowStep, self._windowWidth)
                f.write(e)
                logging.error(e)
                timerForPreFolding.stop()
                raise Exception(e)
            
        elif reference == "end":
            lastPossibleWindowStart = cds.length() - self._windowWidth #+ 1  # disregard lastWindowStart when reference=="end"
            #lastWindowCodonStart = (lastPossibleWindowStart-3)-(lastPossibleWindowStart-3)%3

            #lastPossibleWindowStart = seqLength - windowWidth # + 1  # disregard lastWindowStart when reference=="end"
            requestedWindowStarts = frozenset([x for x in range(lastPossibleWindowStart % windowStep, lastPossibleWindowStart+1, windowStep) if x>=lastWindowStart])

        elif reference == "stop3utr":
            seqLength = cds.length()
            stopCodonPos = cds.CDSlength()
            
            isRequired = [1 if abs(pos-stopCodonPos)<((lastWindowStart//2)*windowStep) else 0 for pos in range(0, seqLength - self._windowWidth, windowStep)]
            requestedWindowStarts = frozenset( compress( range(seqLength), isRequired ) )
            

            #requestedWindowStarts = frozenset(range(lastWindowCodonStart % windowStep, lastWindowCodonStart, windowStep))
            #pass
        else:
            assert(False)

        # First, read available results (for all shuffle-ids) in JSON format
        # Array is indexed by shuffle-id, so results not requested will be represented by None (as will requested items that have no results yet).
        logging.info("DEBUG: requestedShuffleIds (%d items): %s\n" % (len(requestedShuffleIds), requestedShuffleIds))
        existingResults = cds.getCalculationResult2( self._seriesSourceNumber, requestedShuffleIds, True, shuffleType=shuffleType )
        #assert(len(existingResults) >= len(requestedShuffleIds))  # The returned array must be at least as large as the requested ids list
        assert(len(existingResults) == len(requestedShuffleIds))
        logging.info("requestedShuffleIds: %s" % requestedShuffleIds)
        logging.info("existingResults.keys(): %s" % list(existingResults.keys()))
        assert(frozenset(requestedShuffleIds)==frozenset(list(existingResults.keys())))
        #existingResults = [None] * (max(requestedShuffleIds)+1)
        logging.info("DEBUG: existingResults (%d items): %s\n" % (len(existingResults), existingResults))

        # Check for which of the requested shuffle-ids there are values missing
        shuffleIdsToProcess = {}
        for shuffleId, r in list(existingResults.items()):
            if r is None:
                # There are no existing results for shuffled-id n. If it was requested, it should be calculated now (including all windows)
                if shuffleId in requestedShuffleIds:
                    shuffleIdsToProcess[shuffleId] = list(requestedWindowStarts)
                    
                timerForPreFolding.stop()
                
                # ------------------------------------------------------------------------------------
                continue   # TODO - verify this line; should we abort this sequence by throwing????
                # ------------------------------------------------------------------------------------

            logging.info("/// shuffleId r = %d %s" % (shuffleId, r))
            logging.info("r[MFE-profile] %s" % r["MFE-profile"])
            
            # Check the existing results for this shuffle
            alreadyProcessedWindowStarts = frozenset( [i for i,x in enumerate(r["MFE-profile"] ) if x is not None] ) # Get the indices (=window starts) of all non-None values
            missingWindows = requestedWindowStarts - alreadyProcessedWindowStarts # Are there any requested windows that are not already computed?
            if( missingWindows ): 
                shuffleIdsToProcess[shuffleId] = missingWindows

        if( not shuffleIdsToProcess):
            e = "All requested shuffle-ids in (taxId: %d, protId: %s, seqs: %s) seem to have already been processed. Skipping...\n" % (taxId, protId, str(list(zip(seqIds, requestedShuffleIds))) )
            logging.warning(e)
            timerForPreFolding.stop()
            return
        logging.info("DEBUG: shuffleIdsToProcess (%d items): %s\n" % (len(shuffleIdsToProcess), shuffleIdsToProcess))

        logging.info("DEBUG: Before (%d items): %s\n" % (len(existingResults), existingResults))
        # Initialize new results records
        for shuffleId in list(shuffleIdsToProcess.keys()):
            if existingResults[shuffleId] is None:
                logging.info(seqIds)
                logging.info(requestedShuffleIds)
                logging.info(shuffleId)
                thisSeqId = seqIds[ requestedShuffleIds.index(shuffleId) ]
                    
                existingResults[shuffleId] = { "id": "%s/%s/%d/%d" % (taxId, protId, thisSeqId, shuffleId), "seq-crc": None, "MFE-profile": [], "MeanMFE": None, "v": 2, "shuffle-type":shuffleType }
        logging.info("DEBUG: existingResults (%d items): %s\n" % (len(existingResults),existingResults) )
        timerForPreFolding.stop()

        # Load the sequences of all shuffle-ids we need to work on
        # TODO - combine loading of multiple sequences into one DB operation
        for shuffleId, record in list(existingResults.items()):
            if record is None:
                logging.info("DEBUG: skipping empty results record for shuffleId={}".format(shuffleId))
                continue
            timerForPreFolding.start()

            seq = None
            annotatedSeqId = None
            # Get the sequence for this entry
            if( shuffleId < 0 ):
                seq = cds.sequence()
                annotatedSeqId = cds.seqId()
            else:
                seq = cds.getShuffledSeq(shuffleId, shuffleType)
                annotatedSeqId = cds.getShuffledSeqId(shuffleId, shuffleType)

            if( seq is None or (not seq is None and len(seq)==0 )):
                seq2 = cds.getShuffledSeq2( annotatedSeqId )
                seq3 = cds._fetchSequence( annotatedSeqId )
                seq4 = cds._cache.get("%d:seq"%annotatedSeqId)
                if not seq4 is None:
                    del cds._cache["%d:seq"%annotatedSeqId]
                seq5 = cds.getShuffledSeq2( annotatedSeqId )
                e = "Got empty sequence for shuffleId=%d, seqId=%d, taxId=%d, protId=%s, numShuffled=%d, ids[%d:%d]=%s, len(seq2)=%d, len(seq3)=%d, len(seq4)=%d, len(seq5)=%d" % (shuffleId, annotatedSeqId, taxId, protId, len(cds.shuffledSeqIds()), shuffleId-2, shuffleId+2, cds.shuffledSeqIds()[shuffleId-2:shuffleId+2], len(seq2) if not seq2 is None else -1, len(seq3) if not seq3 is None else -1, len(seq4) if not seq4 is None else -1, len(seq5) if not seq5 is None else -1 )
                logging.error(e)
                timerForPreFolding.stop()
                raise Exception(e)

            #
            # Disabled - calculation needn't include the native sequence...
            #
            #if( annotatedSeqId not in seqIds ):
            #    e = "Error: SeqId specified in queue item %s does not match annotated seq-id %d\n" % (itemToProcess, annotatedSeqId)
            #    f.write(e)
            #    f.write("Current shuffle-id: %d\n" % shuffleId)
            #    f.write("Ids in existing results:\n")
            #    for shuffleId, record in enumerate(existingResults):
            #        f.write(" %d) %s\n" % (shuffleId, record['id']))
            #    f.write("Debug info:\n")
            #    f.write("\n".join(cds.getDebugInfo()))
            #    f.write("\n")
            #    f.write("Skipping...\n")
            #    print("Skipping...")
            #    raise Exception(e)

            expectedSeqLength = cds.length()
            if( not expectedSeqLength is None ):
                if( expectedSeqLength != len(seq) ):
                    e = "Warning: taxid=%d, protid=%s, seqid=%d - unexpected length %d (expected: %d)\n" % (taxId, protId, annotatedSeqId, len(seq), expectedSeqLength)
                    f.write(e)
                    logging.error(e)
                    timerForPreFolding.stop()
                    raise Exception(e)

            if( len(seq) < self._windowWidth ):
                # Sequence is shorter than required window; skip
                e = "Warning: skipping sequence because it is shorter than the requested window...\n"
                f.write(e)
                logging.error(e)
                timerForPreFolding.stop()
                raise Exception(e)

            logging.info("DEBUG: Processing item taxId=%d, protId=%s, shuffle=%d (length=%d, %d windows)...\n" % (taxId, protId, shuffleId, len(seq), len(requestedWindowStarts)))

            # TODO - Remove any old value stored in this key?

            # Skip this for now
            # This will be made redundant by completing the "updating" implementation
            #
            #if( cds.isCalculationDone( seriesSourceNumber, shuffleId )):
            #    # Sufficient data seems to exist. Skip...
            #    f.write("Item %s appears to be already completed, skipping..." % itemToProcess)
            #    continue

            logging.info(seq[:50])
            #f.write("\n")

            MFEprofile = record["MFE-profile"]
            #f.write("Profile: %s\n" % MFEprofile)

            # Make sure the profile array contains enough entries for all new windows (and possibly, if windows are non-contiguous, entries between them that we are not going to compute right now)
            if( len(MFEprofile) < max(requestedWindowStarts) ):
                entriesToAdd = max(requestedWindowStarts) - len(MFEprofile) + 1
                MFEprofile.extend( [None] * entriesToAdd )
            assert(len(MFEprofile) >= max(requestedWindowStarts))

            stats = RunningStats()
            stats.extend([x for x in MFEprofile if x is not None])

            timerForPreFolding.stop()
            timerForFolding.start()
            for start in requestedWindowStarts:
                fragment = seq[start:(start+self._windowWidth)]
                assert(len(fragment)==self._windowWidth)

                if self._seriesSourceNumber in (db.Sources.RNAfoldEnergy_SlidingWindow30_v2, db.Sources.RNAfoldEnergy_SlidingWindow40_v2, db.Sources.RNAfoldEnergy_SlidingWindow50_v2):
                    # Calculate the RNA folding energy. This is the computation-heavy part.
                    #strct, energy = RNA.fold(fragment)
                    result = RNAfold_direct(fragment)
                    assert(result <= 0.0)

                elif self._seriesSourceNumber == db.Sources.RNAfoldEnergy_SlidingWindow40_v2_native_temp:
                    # Calculate the RNA folding energy. This is the computation-heavy part.
                    #strct, energy = RNA.fold(fragment)
                    result = RNAfold_direct(fragment, explicitCalculationTemperature = optimalSpeciesGrowthTemperature)
                    assert(result <= 0.0)

                elif self._seriesSourceNumber == db.Sources.GC_content_SlidingWindow40:
                    result = calcWindowGCContent( fragment )
                    assert( isnan(result) or (result >= 0.0 and result <= 1.0) )
                    
                elif self._seriesSourceNumber == db.Sources.Purine_content_SlidingWindow40:
                    result = calcWindowPurineContent( fragment )
                    assert( isnan(result) or (result >= 0.0 and result <= 1.0) )
                    
                elif self._seriesSourceNumber in (db.Sources.StopCodon_content_SlidingWindow30, db.Sources.StopCodon_content_SlidingWindow40, db.Sources.StopCodon_content_SlidingWindow50):
                    result = calcWindowStopCodonContent( fragment, translationTable=genomicTranslationTable, phase=start%3 )
                    assert( result >= 0.0 and result <= 1.0 )

                    
                elif self._seriesSourceNumber == db.Sources.TEST_StepFunction_BeginReferenced:
                    if shuffleId < 0:
                        result = 0
                    else:
                        result = start%50 - 20
                
                elif self._seriesSourceNumber == db.Sources.TEST_StepFunction_EndReferenced:
                    if shuffleId < 0:
                        result = 0
                    else:
                        result = (expectedSeqLength - self._windowWidth - start)%50 - 20

                else:
                    logging.error("Received unknown seriesSourceNumber {}".format(self._seriesSourceNumber))
                    assert(False)
                    
                # Store the calculation result
                #print("%d:%s --> %f" % (taxId, protId, energy))

                stats.push(result)
                MFEprofile[start] = result

            #print("///////////////////  shuffleId={} (len={}) //////////////////////////".format(shuffleId, expectedSeqLength))
            if debug:
                prettyPrintProfile(MFEprofile)

            timerForFolding.stop()
            timerForPostFolding.start()

            # Format
            crc = getCrc(seq)
            #result = """{"id":"%s","seq-crc":%d,"MFE-profile":[%s],"MeanMFE":%.6g,v:2}""" % (itemToProcess, crc, ",".join(map(lambda x: "%.3g" % x, MFEprofile)), stats.mean())
            record["seq-crc"] = crc
            record["MFE-profile"] = [round4(x) for x in MFEprofile] # Round items down to save space (these are not exact numbers anyway)
            record["MeanMFE"] = stats.mean()
            
            if reference == "stop3utr":
                record["stop-codon-pos"] = cds.CDSlength()
                
            result = json.dumps(record)

            f.write(result)
            f.write("\n")

            if( not self._debugDoneWriteResults):
                cds.saveCalculationResult2( self._seriesSourceNumber, result, annotatedSeqId, False )
                
            timerForPostFolding.stop()

            
        timerForPostFolding.start()
        
        if( not self._debugDoneWriteResults):
            cds.commitChanges()
            
        timerForPostFolding.stop()


def parseTaskDescription(taskDescription):
    parts = []
    if( taskDescription.find('/') == -1 ):
        parts = taskDescription.split(":")
    else:
        parts = taskDescription.split("/")

    # Mandatory fields
    taxId, protId, seqId_, shuffleIds_ = parts[:4]
    taxId = int(taxId)
    seqIds = list(map(int, seqId_.split(",")))
    requestedShuffleIds = list(map(int, shuffleIds_.split(",")))
    assert(len(requestedShuffleIds) < 50 ) # Maximum number of shuffle-ids allowed to be combined into a single work record
    assert(seqIds)
    assert(len(seqIds)==len(requestedShuffleIds))

    # Optional fields
    lastWindowStart = 150
    windowStep = 1
    if( len(parts)>4 ):
        lastWindowStart = int(parts[4])
        #assert(lastWindowStart >= 150)
        windowStep = int(parts[5])
        assert(windowStep > 0 and windowStep <= 30)

    windowRef = "begin"
    if( len(parts)>6 ):
        windowRef = parts[6]
    
    shuffleType = db.Sources.ShuffleCDSv2_python
    if( len(parts)>7 ):
        shuffleType = int(parts[7])

    return (taxId, protId, seqIds, requestedShuffleIds, lastWindowStart, windowStep, windowRef, shuffleType)



rl = RateLimit(60)

def calculateTaskForMissingWindowsForSequence(seriesSourceNumber, taskDescription, debug=False):
    config.initLogging()
    logging.warning("Processing task %s" % taskDescription)
    print(taskDescription)
    #def __init__(self, windowWidth=40, logfile=None, debugDoneWriteResults=False, computationTag="rna-fold-window-40-0", seriesSourceNumber=db.Sour
    (taxId, protId, seqIds, requestedShuffleIds, lastWindowStart, windowStep, windowRef, shuffleType) = parseTaskDescription(taskDescription)

    windowWidth = db.getWindowWidthForComputationTag( seriesSourceNumber )
        
    firstWindowStart = 0

    if seriesSourceNumber in (db.Sources.RNAfoldEnergy_SlidingWindow40_v2, db.Sources.RNAfoldEnergy_SlidingWindow30_v2, db.Sources.RNAfoldEnergy_SlidingWindow50_v2, db.Sources.RNAfoldEnergy_SlidingWindow40_v2_native_temp, db.Sources.TEST_StepFunction_BeginReferenced, db.Sources.TEST_StepFunction_EndReferenced, db.Sources.GC_content_SlidingWindow40, db.Sources.Purine_content_SlidingWindow40, db.Sources.StopCodon_content_SlidingWindow30, db.Sources.StopCodon_content_SlidingWindow50, db.Sources.StopCodon_content_SlidingWindow50 ):
        f = CalculateSlidingWindowRandomizedComparisonSeries(windowWidth, seriesSourceNumber=seriesSourceNumber)
    else:
        raise Exception("Got invalid sourceSeries: {}".format(seriesSourceNumber))
    
    try:
        #def calculateMissingWindowsForSequence(self, taxId, protId, seqIds, requestedShuffleIds, firstWindow, lastWindowStart, windowStep, reference="begin"):
        ret = f.calculateMissingWindowsForSequence(taxId, protId, seqIds, requestedShuffleIds, firstWindowStart, lastWindowStart, windowStep, windowRef, shuffleType, debug=debug)
    except Exception as e:
        logging.error("calculateMissingWindowsForSequence() caught exception")
        logging.error(e)
        logging.error(taskDescription)
        raise
        

    if useProfiling and rl():  # display performance timers
        pre  = timerForPreFolding.stats()[0]
        fold = timerForFolding.stats()[0]
        post = timerForPostFolding.stats()[0]
        logging.warning("Performance timers: Pre: %.4gs; Folding: %.4gs; Post: %.4gs; Total: %.4gs" % (pre, fold, post, pre+fold+post))
    
    return taskDescription


"""
Main function for stand-alone worker process -- read and process queued items from redis queue
"""            
def standaloneMainWithRedisQueue():
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--computation-tag")
    argsParser.add_argument("--limit-time")
    argsParser.add_argument("--dry-run", type=parseOption(frozenset(("no", "yes")), "dry-run") )
    args = argsParser.parse_args()

    # Command-line arguments
    computationTag = args.computation_tag
    if( computationTag.find(':') != -1 ): raise Exception("computation tag cannot contain ':' (should be compatible with redis key names)")
    # e.g. rna-fold-0

    runUntil = None
    if( args.limit_time ):
        hours, mins = list(map(int, args.limit_time.split(':')))
        assert(mins<60)
        runUntil = RunUntil(timedelta(0, 0, 0, 0, mins, hours))


    # Configuration
    #queueTag = "queue:tag:awaiting-%s:members" % computationTag
    #seqLengthKey = "CDS:taxid:%d:protid:%s:length-nt"
    seriesSourceNumber = db.Sources.RNAfoldEnergy_SlidingWindow40_v2
    windowWidth = 40
    windowStep = 1


    myWorkerKey = createWorkerKey(computationTag)

    updateJobStatus_ItemCompleted( myWorkerKey )

    
    with gzip.open('worker_%s_%d_log.gz' % (subprocess.check_output('hostname')[:-1], os.getpid()), 'wb') as f:
        print("My worker-key is: %s" % myWorkerKey )
        f.write("My worker-key is: %s\n" % myWorkerKey )

        rnafold = CalculateSlidingWindowRandomizedComparisonSeries(windowWidth, f, False, computationTag, seriesSourceNumber)

        for itemToProcess in QueueSource(computationTag):
            parts = []
            if( itemToProcess.find('/') == -1 ):
                parts = itemToProcess.split(":")
            else:
                parts = itemToProcess.split("/")

            # Mandatory fields
            taxId, protId, seqId_, shuffleIds_ = parts[:4]
            taxId = int(taxId)
            seqIds = frozenset(list(map(int, seqId_.split(","))))
            requestedShuffleIds = list(map(int, shuffleIds_.split(",")))
            assert(len(requestedShuffleIds) < 50 ) # Maximum number of shuffle-ids allowed to be combined into a single work record

            # Optional fields
            lastWindowStart = 150
            windowStep = 1
            if( len(parts)>4 ):
                lastWindowStart = int(parts[4])
                #assert(lastWindowStart >= 150)
                windowStep = int(parts[5])
                assert(windowStep > 0 and windowStep <= 30)

            try:
                rnafold.calculateMissingWindowsForSequence(taxId, protId, seqId_, shuffleIds_, 0, lastWindowStart, windowStep, "begin", f)
                updateJobStatus_ItemCompleted( myWorkerKey, taxId )
                
            except Exception as e:
                logging.error("Exception thrown by calculateMissingWindowsForSequence():")
                logging.error(e)
                print(e)


            if( not runUntil is None):
                if( runUntil.isExpired() ):
                    f.write("Time limit reached, exiting...\n")
                    break

        f.write("Done!\n\n")
        
    return 0
    

    

if __name__=="__main__":
    import sys
    #sys.exit(standaloneMainWithRedisQueue())
    #testTask = "436017/ABO99737/7086688,20745703,7868392,20366322,7856723,7875007,21062723,21062724,21062725,21062726,21062727,21062728,21062729,21062730,21062731,21062732,21062733/-1,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160/780/10"
    #  shuffleId=150, seqId=21063218, taxId=436017, protId=ABO96972
    #testTask = "436017/ABO96972/21063218/150/780/10"
    #testTask = "436017/ABO94655/7087318,8008675,7880166,20829673,20600299,20676590,21064374,21064375,21064376,21064377,21064378,21064379,21064380,21064381,21064382,21064383,21064384/-1,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160/350/10"
    #testTask = "436017/ABP00033/7076680,20044248,20045786,7638494,20042209,20129253,21064286,21064287,21064288,21064289,21064290,21064291,21064292,21064293,21064294,21064295,21064296/-1,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160/640/10"
    #testTask = "436017/ABP00428/7087197,20371424,20676675,20750822,20600300,7810032,21064946,21064947,21064948,21064949,21064950,21064951,21064952,21064953,21064954,21064955,21064956/-1,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160/370/10"
    #testTask = "578462/KNE54425/63247003,63247004,63247005,63247006,63247007,63247008,63247009,63247010,63247011,63247012,63247013,63247014,63247015,63247016,63247017,63247018,63247019,63247020,63247021,63247022/0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/1990/10/begin/12"
    #testTask = "553190/ADB14681/47037871,62847903,62847904,62847905,62847906,62847907,62847908,62847909,62847910,62847911,62847912,62847913,62847914,62847915,62847916,62847917,62847918,62847919,62847920,62847921,62847922/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/410/10/begin/11"
    #testTask = "553190/ADB13628/47037529,62847923,62847924,62847925,62847926,62847927,62847928,62847929,62847930,62847931,62847932,62847933,62847934,62847935,62847936,62847937,62847938,62847939,62847940,62847941,62847942/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/920/10/begin/11"
    #testTask = "553190/ADB13628/47037529,62847923,62847924,62847925,62847926,62847927,62847928,62847929,62847930,62847931,62847932,62847933,62847934,62847935,62847936,62847937,62847938,62847939,62847940,62847941,62847942/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/0/10/end/11"
    #testTask = "553190/ADB14512/47037823,62847943,62847944,62847945,62847946,62847947,62847948,62847949,62847950,62847951,62847952,62847953,62847954,62847955,62847956,62847957,62847958,62847959,62847960,62847961,62847962/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/560/10/begin/11"
    testTask = "553190/ADB14512/47037823,62847943,62847944,62847945,62847946,62847947,62847948,62847949,62847950,62847951,62847952,62847953,62847954,62847955,62847956,62847957,62847958,62847959,62847960,62847961,62847962/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/0/10/end/11"
    #testTask = "553190/ADB14651/47038128,62847963,62847964,62847965,62847966,62847967,62847968,62847969,62847970,62847971,62847972,62847973,62847974,62847975,62847976,62847977,62847978,62847979,62847980,62847981,62847982/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/380/10/begin/11"
    #testTask = "553190/ADB14158/47038273,62847983,62847984,62847985,62847986,62847987,62847988,62847989,62847990,62847991,62847992,62847993,62847994,62847995,62847996,62847997,62847998,62847999,62848000,62848001,62848002/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/650/10/begin/11"
    #testTask = "553190/ADB13518/47037530,62848003,62848004,62848005,62848006,62848007,62848008,62848009,62848010,62848011,62848012,62848013,62848014,62848015,62848016,62848017,62848018,62848019,62848020,62848021,62848022/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/1000/10/begin/11"
    #testTask = "553190/ADB13855/47037679,62848023,62848024,62848025,62848026,62848027,62848028,62848029,62848030,62848031,62848032,62848033,62848034,62848035,62848036,62848037,62848038,62848039,62848040,62848041,62848042/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/1000/10/begin/11"
    #testTask = "553190/ADB14495/47037099,62848043,62848044,62848045,62848046,62848047,62848048,62848049,62848050,62848051,62848052,62848053,62848054,62848055,62848056,62848057,62848058,62848059,62848060,62848061,62848062/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/660/10/begin/11"
    #testTask = "553190/ADB14646/47038172,62848063,62848064,62848065,62848066,62848067,62848068,62848069,62848070,62848071,62848072,62848073,62848074,62848075,62848076,62848077,62848078,62848079,62848080,62848081,62848082/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/950/10/begin/11"
    #testTask = "553190/ADB14057/47037869,62848083,62848084,62848085,62848086,62848087,62848088,62848089,62848090,62848091,62848092,62848093,62848094,62848095,62848096,62848097,62848098,62848099,62848100,62848101,62848102/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/890/10/begin/11"
    #testTask = "553190/ADB13694/47037560,62848103,62848104,62848105,62848106,62848107,62848108,62848109,62848110,62848111,62848112,62848113,62848114,62848115,62848116,62848117,62848118,62848119,62848120,62848121,62848122/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/360/10/begin/11"
    testTask = "9606/ENST00000374610/7890,41638,41639,41640,41641,41642,41643,41644,41645,41646,41647,41648,41649,41650,41651,41652,41653,41654,41655,41656,41657/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/310/10/begin/11"

    #['511145/AAC74177/2293,6206,6207,6208,6209,6210,6211,6212,6213,6214,6215,6216,6217,6218,6219,6220,6221,6222,6223,6224,6225/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC74606/1662,6226,6227,6228,6229,6230,6231,6232,6233,6234,6235,6236,6237,6238,6239,6240,6241,6242,6243,6244,6245/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC74726/2517,6246,6247,6248,6249,6250,6251,6252,6253,6254,6255,6256,6257,6258,6259,6260,6261,6262,6263,6264,6265/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC74763/1403,6266,6267,6268,6269,6270,6271,6272,6273,6274,6275,6276,6277,6278,6279,6280,6281,6282,6283,6284,6285/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC74765/2130,6286,6287,6288,6289,6290,6291,6292,6293,6294,6295,6296,6297,6298,6299,6300,6301,6302,6303,6304,6305/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC77329/1495,6306,6307,6308,6309,6310,6311,6312,6313,6314,6315,6316,6317,6318,6319,6320,6321,6322,6323,6324,6325/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC74631/327,6326,6327,6328,6329,6330,6331,6332,6333,6334,6335,6336,6337,6338,6339,6340,6341,6342,6343,6344,6345/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC74786/3236,6346,6347,6348,6349,6350,6351,6352,6353,6354,6355,6356,6357,6358,6359,6360,6361,6362,6363,6364,6365/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC75485/141,6366,6367,6368,6369,6370,6371,6372,6373,6374,6375,6376,6377,6378,6379,6380,6381,6382,6383,6384,6385/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC74590/2172,6386,6387,6388,6389,6390,6391,6392,6393,6394,6395,6396,6397,6398,6399,6400,6401,6402,6403,6404,6405/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC74040/2563,6406,6407,6408,6409,6410,6411,6412,6413,6414,6415,6416,6417,6418,6419,6420,6421,6422,6423,6424,6425/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20', '511145/AAC73189/1145,6426,6427,6428,6429,6430,6431,6432,6433,6434,6435,6436,6437,6438,6439,6440,6441,6442,6443,6444,6445/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/200/1/stop3utr/20']

    testTask = '511145/AAC77323/1919,64847,64848,64849,64850,64851,64852,64853,64854,64855,64856,64857,64858,64859,64860,64861,64862,64863,64864,64865,64866/-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19/600/1/stop3utr/22'

    #calculateMissingWindowsForSequence(801, testTask)
    calculateTaskForMissingWindowsForSequence(102, testTask, debug=True)
    #calculateTaskForMissingWindowsForSequence(210, testTask, debug=True)
    sys.exit(0)
else:
    # TODO - define args when used as a module
    args = {}
