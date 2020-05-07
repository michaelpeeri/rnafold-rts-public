# Examine data in the "pending updates" table (sequence_series2_updates)
# Try to identify errors, inconsistencies and duplications
# Print stastics on the contents of the updates table
# 
from builtins import input
from builtins import map
from builtins import str
from builtins import zip
from builtins import range
from builtins import object
from collections import Counter
from distutils.util import strtobool 
import argparse
from hashlib import md5
import textwrap
from itertools import zip_longest
from data_helpers import seriesUpdatesSource, splitLongSequenceIdentifier, CDSHelper
from rate_limit import RateLimit
import mysql_rnafold as db
from sqlalchemy import sql
from sqlalchemy.sql.expression import func
from codon_randomization import printCounterAsHistogram


argsParser = argparse.ArgumentParser()
argsParser.add_argument("--verbose", type=int, default=0)
argsParser.add_argument("--groupSize", type=int, default=800)
argsParser.add_argument("--find-duplicates", action="store_true", default=False)
argsParser.add_argument("--find-invalid-records", type=lambda x: bool(strtobool(x)), default=True)  # Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa#comment74363171_15008758
argsParser.add_argument("--perform-delete",  action="store_true", default=False)
argsParser.add_argument("--auto-confirm",  action="store_true", default=False)
argsParser.add_argument("--start-from-id",   type=int, default=0)
argsParser.add_argument("--stop-at-id",   type=int, default=-1)
#argsParser.add_argument("--series-source", type=int, default=db.Sources.RNAfoldEnergy_SlidingWindow40_v2)
args = argsParser.parse_args()

rl = RateLimit(30)


total = 0
class ErrorTypes(object):
    MergeIdentityErrorCount = 1
    UpdateRecordIdentityMismatch = 2
    OriginalRecordIdentityMismatch = 3
    UpdateRecordFormatError = 4
    UpdateProfileTooShort = 5
    ExistingProfileValueCorrupted = 6
    UpdateProfileIsIdentical = 7
    SeqCRCchanged = 8
    IdentifiterInsideRecordChanged = 9
    ConflictingOrDuplicateUpdateRecords = 10
    Information_UpdatedRecord = 1001
    Information_NewRecord = 1002

err = Counter()
recordsByTaxId = Counter()
badUpdateRecords = set()
badOriginalRecords = set()

def extractRecordId(longIdentifier):
    return longIdentifier.split(":")
    

def allUpdatesSource(series=None):

    A = db.sequence_series2_updates.alias(name="A")
    #B = db.sequence_series2_updates.alias(name="B")

    maxIndex = None
    
    if series is None:
        maxIndex = db.connection.execute( sql.select((func.max(A.c.dummy_id),)) ).fetchone()[0]
    else:
        maxIndex = db.connection.execute( sql.select((func.max(A.c.dummy_id),)).where(A.c.source==series) ).fetchone()[0]
        
    if maxIndex is None:
        print("No updates found")
        return
    
    print("Total update records: %d" % maxIndex)
    #maxIndex = 4949
    groupSize = 5000

    if args.stop_at_id > 0:
        maxIndex = args.stop_at_id

    total = 0
    for fromId in range(args.start_from_id, maxIndex, groupSize):
        q = None
        if series is None:
            q = sql.select(( A.c.dummy_id, A.c.sequence_id, func.length(A.c.content))).where(
                sql.and_(
                    #                A.c.dummy_id < B.c.dummy_id,
                    #                A.c.sequence_id == B.c.sequence_id,
                    #                A.c.source == B.c.source,
                    sql.between(A.c.dummy_id, fromId, min(fromId+groupSize-1, maxIndex))
                )).order_by(A.c.dummy_id)
        else:
            q = sql.select(( A.c.dummy_id, A.c.sequence_id, func.length(A.c.content))).where(
                sql.and_(
                    #                A.c.dummy_id < B.c.dummy_id,
                    #                A.c.sequence_id == B.c.sequence_id,
                    A.c.source == series,
                    sql.between(A.c.dummy_id, fromId, min(fromId+groupSize-1, maxIndex))
                )).order_by(A.c.dummy_id)

        #print("Executing query for range(%d, %d)..." % (fromId, fromId+groupSize))
        result = db.connection.execute( q )
        pairRecords = result.fetchall()

        for r in pairRecords:
            yield r
        
        total += len(pairRecords)

    print("Total: %d" % total)
    

def allUpdatesSeriesSource():
    A = db.sequence_series2_updates.alias(name="A")

    series = db.connection.execute( sql.select((func.distinct(A.c.source),)).where(
        A.c.dummy_id > args.start_from_id) ).fetchall()
    # yield 505 # test only
    
    for s in [x[0] for x in series]:
        yield s

def allUpdatesSequenceIdsSource(series=None, chunkSize=1000):
    sequenceIds = []
    for (dummyId, sequenceId, contentLength) in allUpdatesSource(series):
        sequenceIds.append(sequenceId)

        if len(sequenceIds) >= chunkSize:
            yield sequenceIds
            sequenceIds = []
            
    if len(sequenceIds)>0:
        yield sequenceIds

def allSequenceTypesSourceForUpdates(series=None):
    sequenceTypes = Counter()

    count = 0
    for sequenceIds in allUpdatesSequenceIdsSource(series):
        #results = db.connection.execute( sql.select((func.distinct(db.sequences2.c.source),)).select_from(db.sequences2).where(
        #    db.sequences2.c.id.in_( sequenceIds )
        #) ).fetchall()
        
        results = db.connection.execute( sql.select((db.sequences2.c.source, func.count(db.sequences2.c.id).label('count'))).select_from(db.sequences2).where(
            db.sequences2.c.id.in_( sequenceIds )
        ).group_by( db.sequences2.c.source ) ).fetchall()

        sequenceTypes += Counter(dict([(x[0],x[1]) for x in results]))
        
        count += len(sequenceIds)
    print("Debug: allSequenceTypesSourceForUpdates: processed {} rows".format(count))

    return(sequenceTypes)
    
    
"""
Return redundant update records, i.e., those updating the same series (computation) for the same sequence
"""
def findDuplicateUpdates():
    recordsToDelete = set()

    prevRecords = {}

    # Iterate of all duplicate pairs, i.e., pairs of update records that apply to the same (sequence_id, source)
    for series in allUpdatesSeriesSource():

        print("Processing all updates for series %d..." % series)
        
        for (idB, sequenceId, lengthB) in allUpdatesSource(series):

            if not sequenceId in prevRecords:
                prevRecords[sequenceId] = (idB, lengthB)
                continue

            (idA, lengthA) = prevRecords[sequenceId]
            assert(idA < idB)


            recordToDelete = None

            # Try to delete the smaller record (by compressed byte size) - since we are about to destroy data, try to destroy the least
            # if both records have the same size, delete the newest
            # 
            # Note: since any pair may be part of a larger group of duplicate records, this rule must define a strong ordering on the members of the
            #       group. That way, only one member (the "largest" one) will be maintained after a single iteration.
            #
            if lengthA is None:
                if lengthB is None:
                    recordToDelete = idB
                else: # lengthA is None and lengthB is not None:
                    recordToDelete = idA
            elif lengthB is None:
                recordToDelete = idB
            elif lengthA is None and lengthB is None:
                recordToDelete = idB
            elif lengthA > lengthB:
                recordToDelete = idB
            elif lengthB > lengthA:
                recordToDelete = idA
            else:
                recordToDelete = idB

            if( recordToDelete==idA ):
                print("Found duplicate pair: X%d %d" % (idA, idB))
            elif( recordToDelete==idB):
                print("Found duplicate pair: %d X%d" % (idA, idB))
            else:
                assert(False)

            recordsToDelete.add(recordToDelete)

    return recordsToDelete
        

        
if( args.find_duplicates ):
    duplicates = findDuplicateUpdates()
    print(len(duplicates))
    duplicatesList = sorted(list(duplicates))
    print(md5(bytes( str(duplicatesList), encoding="ascii" )).hexdigest())
    badUpdateRecords.update(duplicates)
    if len(duplicates) > 0:
        err[ErrorTypes.ConflictingOrDuplicateUpdateRecords] += len(duplicates)

windowsAddedToProfiles = Counter()
windowsAddedToProfiles_DistanceFromStart = Counter()
windowsAddedToProfiles_DistanceFromEnd   = Counter()
windowsAddedToProfiles_FrameRelativeToStart = Counter()
windowsAddedToProfiles_FrameRelativeToEnd   = Counter()
allWindows_FrameRelativeToStart = Counter()
allWindows_FrameRelativeToEnd = Counter()

def printRecordsSideBySide(record1, record2, columnWidth=40):
    allKeys = set(record1.keys())
    allKeys.update(list(record2.keys()))
    
    for key in sorted(allKeys):
        val1 = record1.get(key, None)
        val2 = record2.get(key, None)

        l1 = textwrap.wrap( str(val1), columnWidth )
        l2 = textwrap.wrap( str(val2), columnWidth )

        print("[{}]".format(key))
        formatSpec = "{{:{}}}\t{{:{}}}".format(columnWidth, columnWidth)
        for i1, i2 in zip_longest( l1, l2, fillvalue='' ):
            print(formatSpec.format(i1,i2))

        

for series in allUpdatesSeriesSource():
    if not args.find_invalid_records:     # Skip this (very slow...) test as requested
        break
    
    print("-------------- series {} --------------".format(series))
    for updates in seriesUpdatesSource(calculationId=series, bulkSize=500, startFromId=args.start_from_id, stopAtId=args.stop_at_id):
        total += len(updates)

        for updateRecord, originalRecord in updates:
            # Returns generator for pairs (updatedRecord, originalRecord)
            # updatedRecord  - (sequence_id, decoded_record, dummy_id)
            # originalRecord - (sequence_id, decoded_record)

            assert(len(updateRecord)==3)
            assert((originalRecord is None) or (len(originalRecord)==2))

            if( args.verbose>=4 ):
                print("================================================================")
                print("<<< %s" % str(originalRecord))
                print("---")
                print(">>> %s" % str(updateRecord))

            # Update identity test
            externalId = updateRecord[0]
            internalLongId = updateRecord[1]['id']
            try:
                internalId = splitLongSequenceIdentifier(internalLongId)
                internalSeqId = internalId[2]

                if(internalSeqId != externalId ):
                    err[ErrorTypes.UpdateRecordIdentityMismatch] += 1
                    badUpdateRecords.add(updateRecord[2])
                    if( args.verbose>2 ):
                        print("UpdateRecordIdentityMismatch  (internalSeqId '%d' != externalId '%d')" % (internalSeqId, externalId))

                recordsByTaxId[internalId[0]] += 1

            except Exception as e:
                err[ErrorTypes.UpdateRecordFormatError] += 1
                badUpdateRecords.add(updateRecord[2])
                print(e)

            if not originalRecord is None:
                # Tests requiring an original record

                err[ErrorTypes.Information_UpdatedRecord] += 1
                # Original identity test

                try:
                    identifierFromOriginalRecord = splitLongSequenceIdentifier(originalRecord[1]['id'])
                    seqIdFromOriginalRecord = splitLongSequenceIdentifier(originalRecord[1]['id'])[2]
                except Exception as e:
                    print(e)
                    #msg = "Original record contains invalid identifier '{}' ('{}')".format(originalRecord[1], originalRecord[0])
                    msg = "Original record contains invalid identifier '{}' ('{}')".format(originalRecord[1]['id'], originalRecord[0])
                    print(msg)
                    printRecordsSideBySide(originalRecord[1], updateRecord[1])

                if( originalRecord[0] != seqIdFromOriginalRecord):
                    # Note: This is an error in the original record. How to treat this?
                    err[ErrorTypes.OriginalRecordIdentityMismatch] += 1
                    badOriginalRecords.add(originalRecord[0])
                    if( args.verbose>2 ):
                        print("OriginalRecordIdentityMismatch (originalRecord[0] '%d' != internalId '%d')" % (originalRecord[0], splitLongSequenceIdentifier(originalRecord[1]['id'])[2]))
                        printRecordsSideBySide(originalRecord, {})

                # records match test
                if( updateRecord[0] != originalRecord[0] ):
                    #print(originalRecord)
                    #print(updateRecord)
                    err[ErrorTypes.MergeIdentityErrorCount] += 1
                    badUpdateRecords.add(updateRecord[2])
                    if( args.verbose>2 ):
                        print("MergeIdentityErrorCount")

                if( updateRecord[1]['seq-crc'] != originalRecord[1]['seq-crc'] ):
                    err[ErrorTypes.SeqCRCchanged] += 1
                    badUpdateRecords.add(updateRecord[2])
                    if( args.verbose>2 ):
                        print("SeqCRCchanged")

                if( updateRecord[1]['id'] != originalRecord[1]['id'] ):
                    err[ErrorTypes.IdentifiterInsideRecordChanged] += 1
                    badUpdateRecords.add(updateRecord[2])
                    if( args.verbose>2 ):
                        print("IdentifiterInsideRecordChanged")

                profile0 = originalRecord[1]['MFE-profile']
                profile1 = updateRecord[1]['MFE-profile']
                if( len(profile1) < len(profile0) ):
                    err[ErrorTypes.UpdateProfileTooShort] += 1
                    badUpdateRecords.add(updateRecord[2])
                    if( args.verbose>2 ):
                        print("UpdateProfileTooShort")
                else:
                    #print('-------- 0 --------')
                    #print(len(profile0))
                    #print(profile0)
                    #print('-------- 1 --------')
                    #print(len(profile1))
                    #print(profile1)
                    profile0.extend([None]*(len(profile1)-len(profile0))) # Add 'None's at the end (to allow comparison of new values)
                    hasNewWindows = False
                    numNewWindows = 0

                    cds = CDSHelper(identifierFromOriginalRecord[0], identifierFromOriginalRecord[1] )
                    cdsLength = cds.length()

                    for pos, vs in enumerate(zip( profile0, profile1 )):
                        v0, v1 = vs

                        if( not v1 is None ):
                            allWindows_FrameRelativeToStart.update( (pos%10,) )
                            allWindows_FrameRelativeToEnd.update( ((cdsLength-pos)%10,) )

                        # Note: this check rejects changes to existing windows (though those might be needed at some point)
                        if( (not v0 is None) and
                            ( (v1 is None) or
                              (abs(v0-v1) >= 1e-8) )
                            ):
                            err[ErrorTypes.ExistingProfileValueCorrupted] += 1
                            badUpdateRecords.add(updateRecord[2])
                            if( args.verbose>2 ):
                                print("ExistingProfileValueCorrupted: {} -> {} ({} pos {})".format(v0,v1, updateRecord[1]['id'], pos))

                            break
                        
                        if( (v0 is None) and (not v1 is None) ): # Is this a new value
                            hasNewWindows = True
                            numNewWindows += 1
                            
                            windowsAddedToProfiles_DistanceFromStart.update( (pos,) )
                            windowsAddedToProfiles_DistanceFromEnd.update( (cdsLength-pos,) )
                            windowsAddedToProfiles_FrameRelativeToStart.update( (pos%10,) )
                            windowsAddedToProfiles_FrameRelativeToEnd.update( ((cdsLength-pos)%10,) )

                    windowsAddedToProfiles.update( (numNewWindows,) )
                    if( not hasNewWindows ):  # Did we find any new values?
                        err[ErrorTypes.UpdateProfileIsIdentical] += 1
                        badUpdateRecords.add(updateRecord[2])
                        if( args.verbose>2 ):
                            print("UpdateProfileIsIdentical")

            else:  # no original record
                err[ErrorTypes.Information_NewRecord] += 1
                
                profile1 = updateRecord[1]['MFE-profile']
                newValues = sum([1 for x in profile1 if not x is None])
                windowsAddedToProfiles.update( (newValues,) )


                rawRecordId = updateRecord[1]['id']
                try:
                    recordId = splitLongSequenceIdentifier(rawRecordId)
                            
                except Exception as e:
                    err[ErrorTypes.UpdateRecordFormatError] += 1
                    badUpdateRecords.add(updateRecord[2])
                    print(e)
                    continue

                cds = CDSHelper(recordId[0], recordId[1] )
                cdsLength = cds.CDSlength()

                newPositions = [x[0] for x in enumerate(profile1) if not x[1] is None]

                for pos in newPositions:
                    windowsAddedToProfiles_DistanceFromStart.update( (pos,) )
                    windowsAddedToProfiles_DistanceFromEnd.update( (cdsLength-pos,) )
                    windowsAddedToProfiles_FrameRelativeToStart.update( (pos%10,) )
                    windowsAddedToProfiles_FrameRelativeToEnd.update( ((cdsLength-pos)%10,) )
                

        if(rl()):
            print(total, err, recordsByTaxId)

        # DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #
        #if( total > 90000):
        #    print("DEBUG: Stopped before end!")
        #    break
        # DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #


    
#def printCounterAsHistogram(counter, stops=(0,1,5,20,100,500,1000)):


    
print(total)
print(err)
print(recordsByTaxId)

print("---- Windows added -----")
#print(windowsAddedToProfiles)
printCounterAsHistogram( windowsAddedToProfiles )

print("---- Windows added (position relative to CDS start) -----")
printCounterAsHistogram(windowsAddedToProfiles_DistanceFromStart, stops=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 400, 500, 1000, 1500, 2000))
print("---- Windows added (position relative to CDS end) -----")
printCounterAsHistogram(windowsAddedToProfiles_DistanceFromEnd,   stops=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300, 400, 500, 1000, 1500, 2000))
print("---- Windows added (phase (N=10) relative to CDS start) -----")
printCounterAsHistogram(windowsAddedToProfiles_FrameRelativeToStart,   stops=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
print("---- Windows added (phase (N=10) relative to CDS end) -----")
printCounterAsHistogram(windowsAddedToProfiles_FrameRelativeToEnd,     stops=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
print("---- All windows after update (phase relative to CDS start) -----")
printCounterAsHistogram(allWindows_FrameRelativeToStart,     stops=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
print("---- All windows after update (phase relative to CDS end) -----")
printCounterAsHistogram(allWindows_FrameRelativeToEnd,     stops=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))



print("---- Sequence types ----")
for series in allUpdatesSeriesSource():
    print("Series {}".format(series))
    
    sequenceSources = allSequenceTypesSourceForUpdates(series)
    print("Sequence (shuffle) types: {}".format(sequenceSources))
    

if( not args.perform_delete ):   # Print SQL statements to delete bad records manually (only if we don't delete them)
    print("-- To delete %d invalid records: --" % len(badUpdateRecords))
    print("use rnafold;")
    todelete = list(badUpdateRecords)
    firstItem = 0
    while(True):
        lastItem = min(firstItem+500, len(todelete))
        if( lastItem <= firstItem ):
            break
        print("-- items %d - %d: --" % (firstItem, lastItem))
        print("delete from sequence_series2_updates")
        print("where dummy_id in (%s);" % (", ".join(map(str, todelete[firstItem:lastItem]))))
        print("go")
        firstItem = lastItem
    
if(badOriginalRecords):
    print("WARNING: Found error in original records.")
    print("WARNING: Original records: ")
    print(badOriginalRecords)

print(total, err, recordsByTaxId)

if( badUpdateRecords and args.perform_delete ):

    if args.auto_confirm:
        response="yes"
    else:
        response = input("Enter 'yes' to perform deletion of %d updates (or anything else to skip): " % len(badUpdateRecords))
        
    if( response=="yes" ):
        todelete = list(badUpdateRecords)
        firstItem = 0
        rowsDeleted = 0

        bulkSize = 1000
        
        while(True):
            lastItem = min(firstItem+bulkSize, len(todelete))
            if( lastItem <= firstItem ):
                break
            
            stmt = db.sequence_series2_updates.delete().where( db.sequence_series2_updates.c.dummy_id.in_( todelete[firstItem:lastItem] ) )
            #print(stmt)
            
            result = db.connection.execute( stmt )

            count = result.rowcount
            result.close()
            #count = lastItem-firstItem-1  # test only

            #print("%d <= %d-%d" % (count, lastItem, firstItem))
            assert( count <= lastItem-firstItem )
            rowsDeleted += count
            
            firstItem = lastItem


        print("Rows deleted: %d (expected: %d)" % (rowsDeleted, len(badUpdateRecords)))

