# Enqueue failed entries (i.e. entries missing the calculation results) for repeated processing.
# Scan all CDS entries from a given taxid; Find entries missing a computation result; Re-insert them into the queue for repeated processing
# Input - taxid1,taxid2,taxid3
#       - computationTag
#       - randomFraction
# Example:
# python2 requeue_sequences_missing_energies_for_sliding_window.py --shuffle-type=11  --from-shuffle 0 --to-shuffle 19 --window-step 10 --profile-reference begin --max-num-windows 32 --species 866499,635003,556484,555500,158189,456481,272632,1307761,505682,400682,6669,436017,412133,104782,412030,211586,190485 --series-source 102 --completion-notification True
# 
# TODO: This script will requeue sequences that have already in the queue but haven't been completed yet.
# TODO: Add support for step-size >1
from builtins import str
from builtins import map
from builtins import zip
from builtins import range
import sys
import codecs
import argparse
from random import randint
from collections import Counter
from itertools import compress
import logging
import traceback
import config
import mysql_rnafold as db
from data_helpers import CDSHelper, countSpeciesCDS, getSpeciesName, SpeciesCDSSource, numItemsInQueue
import _distributed
import dask
from store_new_shuffles import storeNewShuffles
from calculate_sliding_window_series import calculateTaskForMissingWindowsForSequence
import notify_pushover

scheduler = _distributed.open()

config.initLogging()

def parseList(conversion=str):
    def convert(values):
        return list(map(conversion, values.split(",")))
    return convert


argsParser = argparse.ArgumentParser()
argsParser.add_argument("--species", type=parseList(int), required=True)
argsParser.add_argument("--computation-tag", default="rna-fold-window-40-0")
argsParser.add_argument("--random-fraction", type=int, default=1)
argsParser.add_argument("--window-step", type=int, default=10)
argsParser.add_argument("--from-shuffle", type=int, default=-1)
argsParser.add_argument("--to-shuffle", type=int, default=20)
argsParser.add_argument("--shuffle-type", type=str, default="")
argsParser.add_argument("--insert-sequences-only", type=bool, default=False)
argsParser.add_argument("--analyze-only", type=bool, default=False)
argsParser.add_argument("--completion-notification", type=bool, default=False)
argsParser.add_argument("--max-num-windows", type=int, default=200)
argsParser.add_argument("--profile-reference", type=str, default="begin")
argsParser.add_argument("--series-source", type=int, default=db.Sources.RNAfoldEnergy_SlidingWindow40_v2)



#argsParser.add_argument("--log", type=str, default=None)
args = argsParser.parse_args()

# command-line arguments
species = args.species
computationTag = args.computation_tag
if( computationTag.find(':') != -1 ): raise Exception("computation tag cannot contain ':' (should be compatible with redis key names)")
# e.g. rna-fold-0
randomFraction = args.random_fraction
windowStep = args.window_step
maxNumWindows = args.max_num_windows
profileReference = args.profile_reference
shuffleType = args.shuffle_type
#defaultMappingType = (db.Sources.ShuffleCDSv2_matlab, db.Sources.ShuffleCDSv2_python)
shuffleTypesMapping = {""                   :db.Sources.ShuffleCDSv2_python,
                       "11"                 :db.Sources.ShuffleCDSv2_python,
                       "ShuffleCDSv2_matlab":db.Sources.ShuffleCDSv2_python,
                       "ShuffleCDSv2_python":db.Sources.ShuffleCDSv2_python,
                       "12"                 :db.Sources.ShuffleCDS_vertical_permutation_1nt,
                       "ShuffleCDS_vertical_permutation_1nt"
                                            :db.Sources.ShuffleCDS_vertical_permutation_1nt,
                       "20"                 :db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation,
                       "21"                 :db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation_Including_Next_CDS,
                       "22"                 :db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation_Including_Next_CDS_Constant_Overlaps
}
shuffleType=shuffleTypesMapping[args.shuffle_type]

#if not args.log is None:
#    numericLevel = getattr(logging, args.log.upper(), None)
#    if not isinstance( numericLevel, int ):
#        raise Exception("Unknown log level {}".format(args.log))
#    print("set logging to {}".format(args.log))
#    logging.basicConfig(level=numericLevel)

# Configuration
#queueKey = "queue:tag:awaiting-%s:members" % computationTag
#nativeCdsSeqIdKey = "CDS:taxid:%d:protid:%s:seq-id"
#seqLengthKey = "CDS:taxid:%d:protid:%s:length-nt"
lastWindowStart = 2000
seriesSourceNumber = args.series_source
# Explicitly list supported computation tags
if seriesSourceNumber not in ( db.Sources.RNAfoldEnergy_SlidingWindow40_v2, db.Sources.RNAfoldEnergy_SlidingWindow40_v2_native_temp, db.Sources.TEST_StepFunction_BeginReferenced, db.Sources.TEST_StepFunction_EndReferenced, db.Sources.GC_content_SlidingWindow40, db.Sources.Purine_content_SlidingWindow40, db.Sources.StopCodon_content_SlidingWindow30, db.Sources.StopCodon_content_SlidingWindow40, db.Sources.StopCodon_content_SlidingWindow50, db.Sources.RNAfoldEnergy_SlidingWindow30_v2, db.Sources.RNAfoldEnergy_SlidingWindow50_v2 ):
    raise Exception("Unsupported value for --series-source: {}".format(seriesSourceNumber))

# determine the window width for this series
windowWidth = db.getWindowWidthForComputationTag( seriesSourceNumber )


expectedNumberOfShuffles = 20
fromShuffle = args.from_shuffle
toShuffle = args.to_shuffle
# TODO: Add support for step-size >1

# Establish DB connections
#r = redis.StrictRedis(host=config.host, port=config.port, db=config.db)
#session = db.Session()


skipped = 0
selected = 0
alreadyCompleted = 0
totalMissingResults = 0

queuedDelayedCalls = []

for taxIdForProcessing in species:
    print("Processing %d sequences for tax-id %d (%s)..."
          % (countSpeciesCDS(taxIdForProcessing),
             taxIdForProcessing,
             getSpeciesName(taxIdForProcessing)))

    stats = Counter()

    # Iterate over all CDS entries for this species
    # TODO - preloading all sequences and results should optimize this
    for protId in SpeciesCDSSource(taxIdForProcessing):

        stats['all-sequences'] += 1

        #protId = codecs.decode(protId)
        # Filtering

        # Only process 1/N of the sequences, selected randomly (N=randomFraction)
        # (if randomFraction==1, all sequences will be processed)
        if( randint(1, randomFraction) != 1 ):
            skipped += 1
            stats['skipped-random-fraction'] += 1
            continue

        # ------------------------------------------------------------------------------------------
        # Exclude some sequences from the calculation
        # ------------------------------------------------------------------------------------------
        
        # Skip sequences with partial CDS annotations
        #if(r.exists("CDS:taxid:%d:protid:%s:partial" % (taxIdForProcessing, protId))):
        #    skipped += 1
        #    continue

        #if( not r.exists(nativeCdsSeqIdKey % (taxIdForProcessing, protId)) ):
        #    skipped +=1
        #    continue

        cds = CDSHelper(taxIdForProcessing, protId)

        seqLength    = cds.length()
        stopCodonPos = cds.CDSlength()
        
        if seqLength is None:
            print("Warning: Could not find CDS length entry for taxid=%d, protid=%s" % (taxIdForProcessing, protId) )
            skipped += 1
            stats['skipped-cds-length-missing'] += 1
            continue

        # Skip sequences with length <40nt (window width)
        if(seqLength < windowWidth + 1 ):
            print("short seq")
            stats['skipped-short-seq'] += 1
            skipped += 1
            continue

        # ------------------------------------------------------------------------------------------
        # Determine which required windows (from each series) don't have results already
        # ------------------------------------------------------------------------------------------

        if profileReference=="begin":
            requiredWindows = list(range(0, min(seqLength - windowWidth + 1, lastWindowStart), windowStep))
            
            if len(requiredWindows) > maxNumWindows:
                requiredWindows = requiredWindows[:maxNumWindows]
                assert(len(requiredWindows) == maxNumWindows)
                
        elif profileReference=="end":
            lastPossibleWindowStart = seqLength - windowWidth # + 1  # disregard lastWindowStart when reference=="end"
            #lastWindowCodonStart = (lastPossibleWindowStart-3)-(lastPossibleWindowStart-3)%3
            
            #requiredWindows = list(range(lastWindowCodonStart % windowStep, lastWindowCodonStart, windowStep))

            requiredWindows = list(range(lastPossibleWindowStart % windowStep, lastPossibleWindowStart+1, windowStep))

            if len(requiredWindows) > maxNumWindows:  # calculate at most N windows from the end (N=maxNumWindows)
                requiredWindows = requiredWindows[-maxNumWindows:]
                assert(len(requiredWindows) == maxNumWindows)
            
            print("seqLength: {} lastPossibleWindowStart: {}".format(seqLength, lastPossibleWindowStart))
            print(requiredWindows)

            # DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY #
            assert(len((" "*seqLength)[lastPossibleWindowStart:]) == windowWidth)
            # DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY #
            
            # DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY #
            #requiredWindows = []
            # DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY ###  DEBUG ONLY #

        elif profileReference=="stop3utr":

            # Up to maxNumWindows windows, split evenly before and after the stop codon
            isRequired = [1 if abs(pos-stopCodonPos)<((maxNumWindows//2)*windowStep) else 0 for pos in range(0, seqLength - windowWidth, windowStep)]
            requiredWindows = list( compress( range(seqLength), isRequired ) )
            
        else:
            assert(False)

        assert(len(requiredWindows) <= maxNumWindows)
            
        requiredShuffles = [-1] # Check that the native profile exists, regardless of the requested range
        requiredShuffles.extend(list(range(fromShuffle, toShuffle+1)))

        existingResults = None
        try:
            missingResults = cds.checkCalculationResultWithWindows( seriesSourceNumber, requiredShuffles, requiredWindows, shuffleType )
        except IndexError as e:
            msg = "Missing sequences for %s, skipping..." % protId
            print(msg)
            logging.error(msg)
            logging.error(str(e))
            skipped += 1
            stats['skipped-missing-seq'] += 1
            continue

        logging.info("missingResults: %s" % missingResults)

        shufflesWithMissingWindows = [requiredShuffles[n] for n,v in enumerate(missingResults) if v ] # get the indices (not shuffle-ids!) of existing shuffles with missing positions
        print("Existing shuffles with missing windows: %s" % shufflesWithMissingWindows)
        completelyMissingShuffles = [requiredShuffles[n] for n,v in enumerate(missingResults) if v is None]
        print("Missing shuffles: %s" % completelyMissingShuffles)

        if completelyMissingShuffles:
            stats['has-some-shuffles-missing'] += 1
            stats['num-shuffles-missing'] += len(completelyMissingShuffles)

        if shufflesWithMissingWindows:
            stats['has-some-windows-missing'] += 1
            stats['num-windows-missing'] += len(shufflesWithMissingWindows)


        if args.analyze_only:
            continue

        # ------------------------------------------------------------------------------------------
        # Submit a single task that will create all missing randomized sequences (for this sequence)
        # ------------------------------------------------------------------------------------------

        ret = None
        if( completelyMissingShuffles ):
            try:
                ret = storeNewShuffles(cds.getTaxId(), cds.getProtId(), completelyMissingShuffles, shuffleType)
                newIds = ret
                #print(ret)

                #ret = scheduler.submit(storeNewShuffles, cds.getTaxId(), cds.getProtId(), completelyMissingShuffles, shuffleType)
                #newIds = ret.result()
                print("Created new seqs:")
                print(list(zip(completelyMissingShuffles, newIds)))

                # reload cds helper data
                del cds
                cds = CDSHelper(taxIdForProcessing, protId)
                print("(done with new seqs)")

            except Exception as e:
                print("Error creating new seqs (taxId={}, protId={})".format(taxIdForProcessing, protId))
                print(e)
                logging.error(e)
                skipped += 1
                continue

            

        if args.insert_sequences_only:
            continue

        # ------------------------------------------------------------------------------------------
        # Submit a tasks for calculating LFE values for all series that have some values missing
        # ------------------------------------------------------------------------------------------

        if profileReference == "begin":
            lastwin = requiredWindows[-1]
        elif profileReference == "end":
            lastwin = requiredWindows[0]
        elif profileReference=="stop3utr":
            lastwin = maxNumWindows
        else:
            assert(False)
            
        if(shufflesWithMissingWindows):
            #cds.enqueueForProcessing(computationTag, shufflesWithMissingWindows, lastwin, windowStep)#

            shuffleIdsToProcess = sorted(shufflesWithMissingWindows + completelyMissingShuffles)
            allSeqIds = cds.shuffledSeqIds(shuffleType)

            def shuffleIdToSeqId(shuffleId):
                if shuffleId==-1:
                    return cds.seqId()
                else:
                    return allSeqIds[shuffleId]
                
            requiredSeqIds = list(map(shuffleIdToSeqId, shuffleIdsToProcess))

            queueItem = "%d/%s/%s/%s/%d/%d/%s/%d" % (cds.getTaxId(), cds.getProtId(), ",".join(map(str, requiredSeqIds)), ",".join(map(str, shufflesWithMissingWindows + completelyMissingShuffles)), lastwin, windowStep, profileReference, shuffleType)
            print(queueItem)

            # To maximize node utilization, we will delay the main part of the calcualtion, the energy calculation, until after
            # we finished creating all necessary sequences (Otherwise, both types of calculations are interleaved and we may
            # be unable to generate enough work when it is needed).
            #
            # An even better alternative might be to interleave submission of both types of tasks, but give the "loading"
            # tasks higher priority.



            if seriesSourceNumber in (db.Sources.RNAfoldEnergy_SlidingWindow40_v2,
                                      db.Sources.RNAfoldEnergy_SlidingWindow40_v2_native_temp,
                                      db.Sources.RNAfoldEnergy_SlidingWindow30_v2,
                                      db.Sources.RNAfoldEnergy_SlidingWindow50_v2,
                                      db.Sources.TEST_StepFunction_BeginReferenced,
                                      db.Sources.TEST_StepFunction_EndReferenced,
                                      db.Sources.GC_content_SlidingWindow40,
                                      db.Sources.Purine_content_SlidingWindow40,
                                      db.Sources.StopCodon_content_SlidingWindow30,
                                      db.Sources.StopCodon_content_SlidingWindow40,
                                      db.Sources.StopCodon_content_SlidingWindow50 ):
                delayedCall = dask.delayed( calculateTaskForMissingWindowsForSequence )(seriesSourceNumber=seriesSourceNumber, taskDescription=queueItem) # create a delayed call for the calculations needed
            else:
                assert(False)
                
            queuedDelayedCalls.append( delayedCall ) # store the call for later submission

            #print("%s: enqueued %d additional results" % (protId, len(shufflesWithMissingWindows)))
            totalMissingResults += len(shufflesWithMissingWindows)
        else:
            print("No pending shuffles, skipping...")
            skipped += 1
            continue

    print("taxId: {} shuffleType: {}".format(taxIdForProcessing, shuffleType) )
    print(stats)


# ------------------------------------------------------------------------------------------
# Process all deferred tasks, and collect the results
# ------------------------------------------------------------------------------------------
        
print("Added %d additional calculations" % totalMissingResults)
print("%d sequences selected, %d skipped, %d already completed (%d total)" % (selected, skipped, alreadyCompleted, selected+skipped))
#print("queue contains %d items" % numItemsInQueue(computationTag))
#print("%d proteins queued" % len(queuedResults))

futures = scheduler.compute(queuedDelayedCalls) # submit all delayed calculations; obtain futures immediately

_distributed.progress(futures) # wait for all calculations to complete
print("\n")

#results = scheduler.gather(futures) # get the results of the completed calcualtions
results = []
errorsCount = 0
completedCount = 0
for f in futures:
    try:
        r = scheduler.gather(f)
        results.append(r)
    except Exception as e:
        results.append(e)
        logging.error("requeue_sequences...: Exception thrown by async function ")
        logging.error(e)
        errorsCount += 1
        
print(results)
# todo -- recover partial results when errors occur
if( futures ):
    print("%d tasks failed (%.3g%%)" % (errorsCount, float(errorsCount)/len(futures)*100))

del scheduler # free connections?

if args.completion_notification:
    if len(species)==1:
        notify_pushover.notify("Done processing %d genome" % len(species))
    else:
        notify_pushover.notify("Done processing %d genomes" % len(species))
