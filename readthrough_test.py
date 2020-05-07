from scipy import io
import numpy as np
import csv


#             +----------> gene-id?
#             |  +-------> 0
#             |  |    +--> position?
#             |  |    |
#a1["RP_ORF"][1][0][0:10]


data_path = "./data/readthrough_Shir/"
measurement_files = ("ribo_MG1655_MOPS_rep1.mat", "ribo_MG1655_MOPS_rep2.mat", "ribo_rich.mat", "WT_rep1.mat", "WT_rep2.mat", "WT_rep3.mat")
metadata_file = "escCol.mat"
id_conversion_file = "./data/Ensembl/Ecoli/identifiers.tsv"
taxId = 511145
readthroughMeasurementWidth = 50
readthroughThreshold = 0.5

def readReadthroughData(data):
    numGenes = data["RP_ORF"].shape[0]

    allDataForORFs = []
    allDataForUTRs = []
    sumsForORFs = []
    sumsForUTRs = []

    for gene in range(numGenes):
        dataForORFs = data["RP_ORF"][gene][0][-readthroughMeasurementWidth:]
        #print(dataForORFs.shape)
        dataFor3UTRs = data["RP_UTR3"][gene][0][:readthroughMeasurementWidth]
        #print(dataFor3UTRs.shape)
        allDataForORFs.extend(dataForORFs.flat)
        allDataForUTRs.extend(dataFor3UTRs.flat)

        sumsForORFs.append( np.mean( dataForORFs ) )
        sumsForUTRs.append( np.mean( dataFor3UTRs ) )

    #print(len(allDataForORFs))
    #print(len(allDataForUTRs))

    sumsForORFs = np.array( sumsForORFs )
    sumsForUTRs = np.array( sumsForUTRs )
    ratios = sumsForUTRs / sumsForORFs

    print("~~")
    print(sumsForORFs.shape)
    print(sumsForUTRs.shape)
    print(np.sum(sumsForORFs[~np.isnan(sumsForORFs)] > 0.0 ))
    
    
    return ( numGenes, np.array(allDataForORFs), np.array(allDataForUTRs), sumsForORFs, sumsForUTRs, ratios  )

def getIdentifiersMapping():
    ret = {}
    with open(id_conversion_file, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            assert(len(row)==3)
            ret[row[1]] = row[0]
            ret[row[2]] = row[0]
    return ret

    


def plotStatistics():

    metadata = io.loadmat("{}{}".format(data_path, metadata_file))
    sourceIdentifiersTable = metadata["gene_id"]

    def getSourceGeneId(idx:int) -> str:
        return sourceIdentifiersTable[idx][0][0]

    #print(metadata["gene_id"].shape)
    #print(metadata["gene_id"][1])
    #print(metadata["gene_id"][100])
    #print(metadata["gene_id"][1000])
    #print(metadata["gene_id"][1020])

    idTable = getIdentifiersMapping()

    


    allData = [io.loadmat("{}{}".format(data_path, fn)) for fn in measurement_files]
    

    RPratios = np.stack( [readReadthroughData(fn)[5] for fn in allData] )
    ORFreads = np.stack( [readReadthroughData(fn)[3] for fn in allData] )
    ORFreads[np.isnan(ORFreads)] = 0.0
    print(ORFreads.shape)
    RPratios_ = RPratios.copy()
    RPratios_[np.isnan(RPratios_)] = 0.0
    RPratios_[np.isinf(RPratios_)] = 0.0
    print("//")
    print(np.min(RPratios[~np.isnan(RPratios)]))
    print(np.max(RPratios[~np.isnan(RPratios)]))
    print(np.min(RPratios_))
    print(np.max(RPratios_))


    for i, fn in enumerate( measurement_files ):
        
        selectedPos = frozenset( np.nonzero(RPratios[i, np.isfinite(RPratios[i,:])]  > readthroughThreshold )[0] )
        selectedNeg = frozenset( np.nonzero(RPratios[i, np.isfinite(RPratios[i,:])] <= readthroughThreshold )[0] )
        print("///////////////////////")
        print(i)
        print("++")
        print( len(selectedPos) )
        print("--")
        print( len(selectedNeg) )

        positiveIdentifiersSourceFmt = frozenset([getSourceGeneId(x) for x in selectedPos])
        negativeIdentifiersSourceFmt = frozenset([getSourceGeneId(x) for x in selectedNeg])
        assert( not positiveIdentifiersSourceFmt.intersection( negativeIdentifiersSourceFmt ) )

        positiveIdentifiersNativeFmt = [idTable.get(x,None) for x in positiveIdentifiersSourceFmt]
        negativeIdentifiersNativeFmt = [idTable.get(x,None) for x in negativeIdentifiersSourceFmt]

            


if __name__=="__main__":
    import sys
    sys.exit(plotStatistics())

