from scipy import io
from scipy.stats import spearmanr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot') # Use the ggplot style
import seaborn as sns
import csv


#             +----------> gene-id?
#             |  +-------> 0
#             |  |    +--> position?
#             |  |    |
#a1["RP_ORF"][1][0][0:10]


data_path = "/tamir1/mich1/termfold/data/readthrough_Shir/"
measurement_files = ("ribo_MG1655_MOPS_rep1.mat", "ribo_MG1655_MOPS_rep2.mat", "ribo_rich.mat", "WT_rep1.mat", "WT_rep2.mat", "WT_rep3.mat")
metadata_file = "escCol.mat"
id_conversion_file = "/tamir1/mich1/termfold/data/Ensembl/Ecoli/identifiers.tsv"
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

        print( data["RP_UTR3"][gene][0].shape )

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

    

def plotDatafileStatistics(data, identifier):
    numGenes, valuesForORFs, valuesForUTRs, sumsForORFs, sumsForUTRs, ratios = readReadthroughData(data)
    #print(valuesForORFs.shape)
    #print(valuesForUTRs.shape)
    
    fig, ax1 = plt.subplots()
    sns.distplot(valuesForORFs)
    plt.savefig( "RP_distribution_{}_ORF.pdf".format(identifier) )
    plt.close(fig)

    fig, ax1 = plt.subplots()
    sns.distplot(valuesForUTRs)
    plt.savefig( "RP_distribution_{}_3UTR.pdf".format(identifier) )
    plt.close(fig)
    
    fig, ax1 = plt.subplots()
    sns.jointplot(x = sumsForORFs, y = sumsForUTRs)
    plt.savefig( "RP_distribution_{}_joint.pdf".format(identifier) )
    plt.close(fig)
    

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
    
    for data, fn in zip(allData, measurement_files):
        plotDatafileStatistics(data, fn)

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

    # Does the "RP ratios" metric correlate between the different experiments?
    rs = spearmanr( RPratios, axis=1, nan_policy="omit" ).correlation

    fig, ax1 = plt.subplots()
    sns.heatmap(rs, annot=True, ax=ax1)
    plt.savefig( "RP_distribution_spearman.pdf" )
    plt.close(fig)

    print(RPratios[0,:].shape)
    #qs  = np.quantile( RPratios_, 0.90, axis=1 )
    #qs3 = np.quantile( RPratios_, 0.95, axis=1 )
    #qs3 = np.quantile( RPratios_, 0.99, axis=1 )
    #print(qs)
    #print(qs2)
    #print(qs3)
    #for t in (0.1, 0.2, 0.3, 0.8, 0.9, 0.99, 0.999):
    #    print( np.quantile( RPratios_, t, axis=1 ) )

    #tt1 = np.quantile( RPratios_, 0.985, axis=1 )

    #selectedPos = np.any( (RPratios_.T >  tt1), axis=1 )
    #selectedNeg = np.all( (RPratios_.T <= tt1), axis=1 ) & np.any(ORFreads > 0.0, axis=0)

    from data_helpers import SpeciesCDSSource, setCDSProperty

    for i, fn in enumerate( measurement_files ):
        
        selectedPos = frozenset( np.nonzero(RPratios[i, np.isfinite(RPratios[i,:])]  > readthroughThreshold )[0] )
        selectedNeg = frozenset( np.nonzero(RPratios[i, np.isfinite(RPratios[i,:])] <= readthroughThreshold )[0] )
        print("///////////////////////")
        print(i)
        # print("++")
        # print( len(selectedPos) )
        # print("--")
        # print( len(selectedNeg) )

        positiveIdentifiersSourceFmt = frozenset([getSourceGeneId(x) for x in selectedPos])
        negativeIdentifiersSourceFmt = frozenset([getSourceGeneId(x) for x in selectedNeg])
        assert( not positiveIdentifiersSourceFmt.intersection( negativeIdentifiersSourceFmt ) )

        positiveIdentifiersNativeFmt = [idTable.get(x,None) for x in positiveIdentifiersSourceFmt]
        negativeIdentifiersNativeFmt = [idTable.get(x,None) for x in negativeIdentifiersSourceFmt]


        # good = 0
        # bad = 0
        # out = []
        # for pos in selectPosIndices:
        #     sourceIds = metadata["gene_id"][pos]
        #     x = sourceIds[0][0]
        #     print(x)
        #     if x in idTable:
        #         good += 1
        #         out.append(idTable[x])
        #     else:
        #         bad += 1
        # print("good={} bad={}".format(good, bad))


        countMarkedPositive = 0
        countMarkedNegative = 0

        for protId in SpeciesCDSSource(taxId):
            valForProt = None
            
            if protId in positiveIdentifiersNativeFmt:
                valForProt = "1"
                countMarkedPositive += 1
                
            elif protId in negativeIdentifiersNativeFmt:
                valForProt = "0"
                countMarkedNegative += 1

            if not valForProt is None:
                setCDSProperty(taxId, protId, "readthrough-v2.ex{}".format(i), valForProt, overwrite=True)

        print( countMarkedPositive )
        print( countMarkedNegative )
        

    #def setSpeciesProperty(taxId, propName, propVal, source, overwrite=True):
    #for protId in SpeciesCDSSource(taxId):
    #    pass
    #    #setCDSProperty(taxId, protId, "readthrough-v1", "1" if protId in positiveSet else "0", overwrite=True)


            


if __name__=="__main__":
    import sys
    sys.exit(plotStatistics())
