from builtins import zip
from builtins import range
import logging
import mysql_rnafold as db
from data_helpers import CDSHelper, AddShuffledSequences, SpeciesCDSSource, getSpeciesTranslationTable
from codon_randomization import SynonymousCodonPermutingRandomization, VerticalRandomizationCache, NucleotidePermutationRandomization, CDSand3UTRRandomization, CDSand3UTRRandomizationIncludingNextCDS
from genome_model import getGenomeModelFromCache

def storeRandomizedSequences(cds, seqs, shuffleIds, shuffleType):
    assert( len(seqs)==len(shuffleIds) )
    addseqs = AddShuffledSequences( cds.getTaxId(), cds.getProtId() )

    newSequenceIds = []
    for seq, shuffleId in zip(seqs, shuffleIds):
        newSequenceId = addseqs.addSequence( shuffleId, seq, shuffleType )
        newSequenceIds.append(newSequenceId)

    return newSequenceIds


# Return randomized sequences for the specified native sequence object
def createRandomizedSeqs(cds, newShuffleIds, shuffleType=db.Sources.ShuffleCDSv2_python):
   
    shuffler = SynonymousCodonPermutingRandomization(cds.getTranslationTable())

    nativeSeq = cds.sequence()
    #print(nativeSeq[:10])

    newShuffles = []
    for shuffleId in newShuffleIds:
        totalPermutationsCount, identity, newseq = None, None, None

        try:
            totalPermutationsCount, identity, newseq = shuffler.randomize(nativeSeq)
        except Exception as e:
            print(e)
            raise
        
        assert((identity <= 1.0) and (identity > 0.0))
        
        if(identity > 0.95):
            print("Warning: Identity of randomized sequence is high - %.3g%% (length=%d nt, total permutations=%.2g)" % (identity*100.0, len(newseq), totalPermutationsCount))
        
        if(totalPermutationsCount < 500):
            raise Exception("Low number of possible permutations %.2g (length=%d nt, identity=%.3g%%)" % (totalPermutationsCount, len(newseq), identity*100.0))
        newShuffles.append( newseq )
    
    return newShuffles


# Return randomized sequences for the specified native sequence object
def createRandomizedSeqs_CDS_with_3UTR(cds, newShuffleIds, shuffleType=db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation, taxId=None):
    #NucleotidePermutationRandomization, CDSand3UTRRandomization
    cdsRand = SynonymousCodonPermutingRandomization(cds.getTranslationTable())
    utrRand = NucleotidePermutationRandomization()

    if shuffleType == db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation:
        shuffler = CDSand3UTRRandomization( cdsRand, utrRand )
        
    elif shuffleType == db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation_Including_Next_CDS:
        shuffler = CDSand3UTRRandomizationIncludingNextCDS( cdsRand, utrRand, taxId=taxId, constantOverlaps=False )
        
    elif shuffleType == db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation_Including_Next_CDS_Constant_Overlaps:
        shuffler = CDSand3UTRRandomizationIncludingNextCDS( cdsRand, utrRand, taxId=taxId, constantOverlaps=True )

    else:
        raise Exception("Unknown shuffleType={}".format(shuffleType))


    genomeModel = getGenomeModelFromCache( cds.getTaxId() )

    nativeSeq = cds.sequence()
    stopCodonPos = cds.CDSlength()
    #print(nativeSeq[:10])

    newShuffles = []
    for shuffleId in newShuffleIds:
        totalPermutationsCount, identity, newseq = None, None, None

        try:
            totalPermutationsCount, identity, newseq = shuffler.randomize( nativeSeq, cds.getProtId() )
        except Exception as e:
            print(e)
            raise

        assert((identity <= 1.0) and (identity > 0.0))
        
        if(identity > 0.95):
            print("Warning: Identity of randomized sequence is high - %.3g%% (length=%d nt, total permutations=%.2g)" % (identity*100.0, len(newseq), totalPermutationsCount))
        
        if(totalPermutationsCount < 500):
            raise Exception("Low number of possible permutations %.2g (length=%d nt, identity=%.3g%%)" % (totalPermutationsCount, len(newseq), identity*100.0))
        newShuffles.append( newseq )
    
    return newShuffles



_caches = {}
    
def getRandomizedSequenceCacheForVerticalPermutations(taxId):
    global _caches

    if (taxId, db.Sources.ShuffleCDS_vertical_permutation_1nt) in _caches:
        cache = _caches[(taxId, db.Sources.ShuffleCDS_vertical_permutation_1nt)]
        
    else:
        # read all native sequences
        protIds = []
        cdss = []
        for protId in SpeciesCDSSource(taxId):
            cds = CDSHelper(taxId, protId)
            
            if( cds.length()%3 != 0 ):
                continue
            
            seq = cds.sequence()
            
            protIds.append(protId)
            cdss.append(seq)
            
        geneticCode = getSpeciesTranslationTable( taxId )
        scpr = SynonymousCodonPermutingRandomization( geneticCode ) 
        randomizer = lambda cdss: scpr.verticalPermutation( cdss )
        cache = VerticalRandomizationCache(shuffleType=db.Sources.ShuffleCDS_vertical_permutation_1nt,
                                           taxId=taxId,
                                           nativeSeqsMap=dict(list(zip(protIds, cdss))),
                                           geneticCode=geneticCode,
                                           randomizer=randomizer )
        _caches[(taxId, db.Sources.ShuffleCDS_vertical_permutation_1nt)] = cache
        print(list(_caches.keys()))

        
    return cache


# Generate and store new randomized sequences for the specified native protein
def storeNewShuffles(taxId, protId, newShuffleIds, shuffleType=db.Sources.ShuffleCDSv2_python, dontStore=False):
    
    cds = CDSHelper(taxId, protId)
    #print(protId)
    
    if shuffleType == db.Sources.ShuffleCDSv2_python:
        return storeRandomizedSequences(cds,
                                        createRandomizedSeqs(cds, newShuffleIds, shuffleType),
                                        newShuffleIds,
                                        shuffleType
        )
    
    elif shuffleType == db.Sources.ShuffleCDS_vertical_permutation_1nt:
        cache = getRandomizedSequenceCacheForVerticalPermutations( taxId )

        seqs = [cache.getShuffledSeq( protId, shuffleId ) for shuffleId in newShuffleIds]
        print(seqs)

        if dontStore: return seqs
        
        return storeRandomizedSequences(cds,
                                        seqs,
                                        newShuffleIds,
                                        shuffleType)
                                
    elif shuffleType == db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation:
        #print("store: before")
        #a = createRandomizedSeqs_CDS_with_3UTR(cds, newShuffleIds, shuffleType)
        #print("store: {}".format(a))

        return storeRandomizedSequences(cds,
                                        createRandomizedSeqs_CDS_with_3UTR(cds, newShuffleIds, shuffleType=shuffleType, taxId=taxId),
                                        newShuffleIds,
                                        shuffleType
        )

    elif shuffleType == db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation_Including_Next_CDS:
        #print("store: before")
        #a = createRandomizedSeqs_CDS_with_3UTR(cds, newShuffleIds, shuffleType)
        #print("store: {}".format(a))

        return storeRandomizedSequences(cds,
                                        createRandomizedSeqs_CDS_with_3UTR(cds, newShuffleIds, shuffleType=shuffleType, taxId=taxId),
                                        newShuffleIds,
                                        shuffleType
        )

    elif shuffleType == db.Sources.ShuffleCDS_synon_perm_and_3UTR_nucleotide_permutation_Including_Next_CDS_Constant_Overlaps:
        #print("store: before")
        #a = createRandomizedSeqs_CDS_with_3UTR(cds, newShuffleIds, shuffleType)
        #print("store: {}".format(a))

        return storeRandomizedSequences(cds,
                                        createRandomizedSeqs_CDS_with_3UTR(cds, newShuffleIds, shuffleType=shuffleType, taxId=taxId),
                                        newShuffleIds,
                                        shuffleType
        )
    
    else:
        raise Exception("Unsupported shuffleType={}".format(shuffleType))
    


if __name__=="__main__":
    import sys
    taxId = int(sys.argv[1])
    storeNewShuffles( taxId, "dummy_protid", list(range(20)), db.Sources.ShuffleCDS_vertical_permutation_1nt, dontStore=True ) # force the creation of 20 sets of randomized sequences for the specified species




    
