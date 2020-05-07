import csv
import numpy as np
from data_helpers import getSpeciesProperty, SpeciesCDSSource, CDSHelper

def parsePaxDbFile(filename, taxId = None):
    abundanceData = {}

    # External protein-ids appear prefixed by the tax-id; this should be removed
    if not taxId is None:
        fixedPrefixToRemove = "{}.".format(taxId)
    else:
        fixedPrefixToRemove = ""
    
    with open(filename, 'r') as csvfile:
        for row in csv.reader(csvfile, delimiter='\t'):
            if not row: continue
            if row[0][0]=='#': continue
            assert( len(row) >= 3 )

            # Extract the protein-id
            protId = row[1]
            if protId[:len(fixedPrefixToRemove)]==fixedPrefixToRemove: # remove common prefix
                protId = protId[len(fixedPrefixToRemove):]

            # Extract the abundance
            protAbundance = float( row[2] )

            # Store the results
            abundanceData[protId] = protAbundance
    return abundanceData


def getSpeciesPaxdbDataFromFile( taxId : int ) -> dict:
    (paxfn, _) = getSpeciesProperty( taxId, 'paxdb-path' )
    if paxfn is None: return {}

    return parsePaxDbFile( paxfn, taxId=taxId )


def convertValuesDictToPercentiles( rawdata : dict ) -> dict:
    if not rawdata: return {}
    k = list(rawdata.keys())
    v = list(rawdata.values())
    vv = np.array( v )
    percentileRanks = np.vectorize( lambda x, a: (x>a).sum()/len(a), excluded=frozenset((1,))  )(vv, vv) # no built-in for percentile ranks...
    assert(vv.shape==percentileRanks.shape)
    return dict(zip(k, percentileRanks))

_paxdbData = {}
def getSpeciesPaxdbData( taxId : int, convertToPercentiles : bool = True ) -> dict:
    ret = _paxdbData.get( (taxId,convertToPercentiles), None )
    if not ret is None: return ret

    newdata = getSpeciesPaxdbDataFromFile( taxId )
    if convertToPercentiles:
        newdata = convertValuesDictToPercentiles( newdata )
        
    _paxdbData[ (taxId,convertToPercentiles) ] = newdata
    return newdata
    
def testSpecies(taxId):
    paData = getSpeciesPaxdbData( taxId )

    countFound = 0
    countNotFound = 0
    
    for protId in SpeciesCDSSource(taxId):
        cds = CDSHelper( taxId=taxId, protId=protId )
        geneId = cds.getGeneId()
        
        if geneId in paData:
            countFound += 1
        else:
            countNotFound += 1

    print("Species: {} -> Found: {} ({:.3}%) Not found: {}".format(taxId, countFound, countFound/(countFound+countNotFound)*100, countNotFound))
    return( countFound, countNotFound)
        
    

def testAll():
    #ret = parsePaxDbFile( '/tamir1/mich1/termfold/data/PaxDB/722438-Mycoplasma_pneumoniae_M129_Kuhner_et_al_Science2009.txt', taxId=722438 )
    #print(ret)
    #allSpeciedWithPaxdbData = frozenset((1148,158878,160490,169963,192222,208964,224308,243159,267671,272623,283166,449447,511145,546414,593117,64091,722438,83332,85962,882,99287))
    allSpeciedWithPaxdbData = frozenset((1148,158878,160490,169963,192222,208964,224308,243159,267671,272623,283166,449447,511145,546414,64091,83332,85962,882,99287,722438,196627,298386))
    
    for taxId in allSpeciedWithPaxdbData:
        testSpecies(taxId)
    return 0
    
if __name__=="__main__":
    import sys
    sys.exit(testAll())
