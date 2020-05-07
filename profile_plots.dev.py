# Read all computed series data from DB; write summary to hdf5 
# Replaces convert_data_for_plotting.py
#
from builtins import map
from builtins import zip
from builtins import range
from builtins import object
from collections import Counter  # Testing only
import argparse
import random
import csv
import numpy as np
import pandas as pd
from math import log10, floor
from tempfile import NamedTemporaryFile
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import NucleotideAlphabet
from scipy.stats import wilcoxon, spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from data_helpers import decompressNucleicSequence, checkSpeciesExist, getSpeciesFileName, allSpeciesSource, getCDSProperty, CDSHelper
from mysql_rnafold import Sources, getWindowWidthForComputationTag
from process_series_data import readSeriesResultsForSpeciesWithSequence, convertResultsToMFEProfiles, sampleProfilesFixedIntervals, profileLength, profileElements, MeanProfile, calcSampledGCcontent, profileEdgeIndex
from mfe_plots import plotMFEProfileWithGC, plotMFEProfileV3, plotXY, scatterPlot, scatterPlotWithColor, plotMFEProfileByPA, plotMFEProfileMultiple, scatterPlotWithKernel, plotMFEProfileForMultipleRandomizations, plot2WayMFEComparison, plotMultipleMFEComparison, plot2dProfile, plot2dProfileAsRidgeplot, plotMultipleDestinations, plotSmoothedLengthEffectData, plotHistogramComparison, plotPAratioVsDistanceHeatmap, plotPAregression
from codonw import readCodonw, meanCodonwProfile
from genome_model import getGenomeModelFromCache
from regulondb_tables import get3primeUtrLength
from paxdb import getSpeciesPaxdbData
from datetime import datetime
from rate_limit import RateLimit
rl = RateLimit(30)


# ------------------------------------------------------------------------------------
# Command-line args

def parseList(conversion=str):
    def convert(values):
        return list(map(conversion, values.split(",")))
    return convert
    
def parseProfileSpec():
    def convert(value):
        o = value.split(':')
        assert(len(o) >= 3 and len(o) <= 4)
        
        o[0] = int(o[0])
        assert(o[0]>0)
        
        o[1] = int(o[1])
        assert(o[1]>0)
        
        assert(o[2]=="begin" or o[2]=="end" or o[2]=="stop3utr")

        if( len(o) == 4 ):
            o[3] = int(o[3])
        else:
            o.append(0)
        
        return (o[0], o[1], o[2], o[3])
    return convert


class PlottingDestination(object):
    def __init__(self, args, shuffleType, taxid, subclass="all", wilcoxonDLFEZero=False, wilcoxonDLFE50_100=False):
        self.args = args
        self.taxid = taxid
        self.shuffleType = shuffleType
        self.subclass = subclass
        self.calcWilcoxonDLFEZero = wilcoxonDLFEZero
        self.calcWilcoxonDLFE50_100 = wilcoxonDLFE50_100

        self.reset()

    def accumulate(self, profileData, seq, fullCDS, positions:dict={}):
        args = self.args

        self.nativeMeanProfile.add( profileData[0,None] )
        self.shuffledMeanProfile.add( profileData[1:] )
        #self.deltaLFEMeanProfile.add( profileData[0,None] - profileData[1:] )


        # Prepare GC profile
        gc = calcSampledGCcontent( seq, args.profile[1] )
        if( gc.size > profileLength(args.profile) ):  # truncate the profile if necessary
            gc = np.resize( gc, (profileLength(args.profile),))
        self.GCProfile.add( np.expand_dims(gc,0) )

        # Prepare percentile mean profiles
        self.shuffled25Profile.add( np.expand_dims( np.percentile(profileData[1:], 25, axis=0), 0) )
        self.shuffled75Profile.add( np.expand_dims( np.percentile(profileData[1:], 75, axis=0), 0) )

        cds_length_nt = len(fullCDS)
        self.cdsLengths.append(cds_length_nt)

        # Basic case:
        #
        # Profile:        |<--------|-------->|
        #
        #          Main-CDS                   Next-CDS
        # |------------------------->    |----------------->
        #
        #
        # Short main-CDS:
        # Profile:        |XXX------|-------->|
        #                     |----->    |----------------->
        # Updstream-pad   |--->
        #
        # Short next-CDS:
        # Profile:        |---------|------XXX|
        # |------------------------->  |-->
        # Downstream-pad                   |-->

        if self.calcWilcoxonDLFEZero or self.calcWilcoxonDLFE50_100:
            #upstreamPaddingLength   = max(0, int(floor(profileLength(args.profile)/2)) - positions['stop-codon'])
            #nextCDSLength = cds_length_nt - positions['stop-codon']-positions['3utr-length']
            #assert(nextCDSLength % 3 == 0)
            #downstreamPaddingLength = max(0, int(floor(profileLength(args.profile)/2)) - nextCDSLength )
            #if upstreamPaddingLength + profileData.shape[1] + downstreamPaddingLength != profileLength(args.profile):
            #    print(1)
            #assert(upstreamPaddingLength + profileData.shape[1] + downstreamPaddingLength == profileLength(args.profile))
            if profileData.shape[1]==200:
                self.wilcoxon = np.concatenate( (self.wilcoxon, profileData[0,None] - profileData[1:,:] ) )
                
            
        


    def reset(self):
        args = self.args
        self.profileLen = profileLength(args.profile)
        
        self.nativeMeanProfile   = MeanProfile( self.profileLen )
        self.shuffledMeanProfile = MeanProfile( self.profileLen )
        #self.deltaLFEMeanProfile = MeanProfile( profileLen )
        self.shuffled25Profile   = MeanProfile( self.profileLen )
        self.shuffled75Profile   = MeanProfile( self.profileLen )
        self.GCProfile           = MeanProfile( self.profileLen )

        if self.calcWilcoxonDLFEZero or self.calcWilcoxonDLFE50_100:
            self.wilcoxon = np.zeros( shape=(0, self.profileLen), dtype=np.float )


        self.cdsLengths = []


    def printDebugInfo(self):
        print("subclass = {}".format(self.subclass))
        print(self.nativeMeanProfile.counts())
        print(self.shuffledMeanProfile.counts())

    def finalize(self):
        # -----------------------------------------------------------------------------
        # Save mean profiles as H5

        # Format (for compatible with plot_xy.py and old convert_data_for_plotting.py:
        #         gc  native  position  shuffled
        # 1    0.451  -4.944         1    -5.886
        # 2    0.459  -5.137         2    -6.069
        # 3    0.473  -5.349         3    -6.262

        args = self.args
        taxid = self.taxid
        shuffleType = self.shuffleType

        xRange = profileElements(args.profile)

        if max(self.nativeMeanProfile.counts()) == 0:
            return
        
        df = pd.DataFrame( {
            "native": self.nativeMeanProfile.value(),
            "shuffled": self.shuffledMeanProfile.value(),
            "gc":self.GCProfile.value(),
            "position": xRange,
            "shuffled25":self.shuffled25Profile.value(),
            "shuffled75":self.shuffled75Profile.value()},
            index=xRange )

        statisticsDF = pd.DataFrame({
            'mean_mean_gc': pd.Series([np.mean(self.GCProfile.value())]),
            'taxid': pd.Series([taxid], dtype='int'),
            'cds_count': pd.Series([len(self.cdsLengths)], dtype='int'),
            'media_cds_length_nt': pd.Series([np.median(self.cdsLengths)])
        })

        if self.calcWilcoxonDLFEZero:
            # perform wilcoxon test on all the deltas at each position
            wilcoxonResults = np.apply_along_axis( wilcoxon, axis=0, arr=self.wilcoxon )
            assert( wilcoxonResults.shape == (2, self.profileLen) )
            # wilcoxonResults[0] = statistics
            # wilcoxonResults[1] = p-values
            wilcoxonSigns = np.sign( np.apply_along_axis( np.mean, axis=0, arr=self.wilcoxon ) ) # store the sign for each position (i.e., if the test is significant, are the values significantly positive or negative?)

            wilcoxonDLFEZero = pd.DataFrame({
                'pos':pd.Series(xRange, dtype='int'),
                'pval':pd.Series(wilcoxonResults[1], dtype='float'),
                'N':pd.Series( np.repeat( self.wilcoxon.shape[0], self.profileLen), dtype='int'),
                'sign':pd.Series(wilcoxonSigns, dtype='int')})

        if self.calcWilcoxonDLFE50_100:
            startIdx = profileElements(args.profile).index(75)
            stopIdx  = profileElements(args.profile).index(99)
            mean50_100 = np.mean( self.wilcoxon[:,startIdx:stopIdx+1] )
            dataForTest = self.wilcoxon - mean50_100
            
            wilcoxonResults = np.apply_along_axis( wilcoxon, axis=0, arr=dataForTest )
            assert( wilcoxonResults.shape == (2, self.profileLen) )

            wilcoxonSigns = np.sign( np.apply_along_axis( np.mean, axis=0, arr=dataForTest ) ) # store the sign for each position (i.e., if the test is significant, are the values significantly positive or negative?)
            
            wilcoxonDLFE50_100 = pd.DataFrame({
                'pos':pd.Series(xRange, dtype='int'),
                'pval':pd.Series(wilcoxonResults[1], dtype='float'),
                'N':pd.Series( np.repeat( dataForTest.shape[0], self.profileLen), dtype='int'),
                'sign':pd.Series(wilcoxonSigns, dtype='int')})
            
        
        if( args.computation_tag == Sources.RNAfoldEnergy_SlidingWindow40_v2 ):
            h5fn = "gcdata_v3_taxid_{}_profile_{}_{}_{}_{}_t{}_{}.h5".format(taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3], shuffleType, self.subclass)
        else:
            h5fn = "gcdata_v3_taxid_{}_profile_{}_{}_{}_{}_t{}_{}_series{}.h5".format(taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3], shuffleType, self.subclass, args.computation_tag)

        # Compression parameters are described here:  http://www.pytables.org/usersguide/libref/helper_classes.html#filtersclassdescr
        # ...and discussed thoroughly in the performance FAQs
        with pd.io.pytables.HDFStore(h5fn, complib="zlib", complevel=1) as store:
            store["df_%d_%d_%d_%s_%d" % (taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3])] = df
            #store["deltas_%d_%d_%d_%s_%d" % (taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3])] = deltasForWilcoxon
            #store["spearman_rho_%d_%d_%d_%s_%d" % (taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3])] = spearman_rho
            store["statistics_%d_%d_%d_%s_%d" % (taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3])] = statisticsDF
            #if( args.codonw ):
            #    store["profiles_spearman_rho_%d_%d_%d_%s_%d" % (taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3])] = dfProfileCorrs
            if self.calcWilcoxonDLFEZero:
                store["wilcoxon_dlfe_%d_%d_%d_%s_%d" % (taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3])] = wilcoxonDLFEZero
            if self.calcWilcoxonDLFE50_100:
                store["wilcoxon_dlfe50_100_%d_%d_%d_%s_%d" % (taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3])] = wilcoxonDLFE50_100            
            #store["transition_peak_wilcoxon_%d_%d_%d_%s_%d" % (taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3])] = guPeakDf
            #store["edge_wilcoxon_%d_%d_%d_%s_%d" % (taxid, args.profile[0], args.profile[1], args.profile[2], args.profile[3])] = edgeWilcoxonDf

            store.flush()

        return df

    def getNativeProfile(self):
        return self.nativeMeanProfile.value()
    
    def getShuffledProfile(self):
        return self.shuffledMeanProfile.value()

    #def getDeltaLFEProfile(self):
    #    return self.deltaLFEMeanProfile.value()

def getSmoothedPropertyEffectOnProfiles(dLFEmatrix, propertyValues, sigma=1.0, valuesRange=range(-50,300,1), valuesPropName='Pos', valuesPropType='int'):
    smoothedLengthEffect = pd.DataFrame( {valuesPropName:pd.Series(dtype=valuesPropType), 'MinDLFE':pd.Series(dtype='float'), 'Density':pd.Series(dtype='float')} )
    
    for l in valuesRange:
        weights = np.exp(-((np.array(propertyValues) - l)**2)/sigma)
        density = sum(weights)
        weights = weights/density
        #np.max(np.ones((3383,200)).T * np.expand_dims(wt, axis=0))
        profile = np.sum( dLFEmatrix * np.expand_dims( weights, axis=0 ).T, axis=0 )
        peak = np.min(profile)
        smoothedLengthEffect = smoothedLengthEffect.append( pd.DataFrame( {valuesPropName:pd.Series([l]), 'MinDLFE':pd.Series([peak]), 'Density':pd.Series([density])} ))

    return smoothedLengthEffect


def percentileRanks( data ):
    return np.vectorize( lambda x, a: (x>a).sum()/len(a), excluded=frozenset((1,))  )(data, data) # no built-in for percentile ranks...
    
    
class ProfilePlot(object):
    def __init__(self, taxId, args):

        self._taxId = taxId
        self._args = args
        
        # pa = {}
        # if( args.pax_db ):
        #     with open(args.pax_db, "rb") as csvfile:
        #         reader = csv.reader(csvfile, delimiter=',')
        #         for row in reader:
        #             rank = float(row[2])
        #             assert(rank >= 0.0 and rank <= 1.0 )
        #             pa[row[0]] = rank

        # Determine the window width
        self.windowWidth = getWindowWidthForComputationTag(args.computation_tag)
                    

    def performPlots(self):
        
        args = self._args
        taxid = self._taxId

        # ------------------------------------------------------------------------------------

        numShuffledGroups = args.num_shuffles
        shuffleTypes = args.shuffle_types
        print("*********** {} ***********".format(shuffleTypes))

        combinedData = {}

        stopCodonFreq = Counter()

        frout = open("out.csv", "wt")
        frdump = open("out_{}.csv".format(taxid), "wt")

        utr3primeLengths = get3primeUtrLength(taxid)

        for shuffleType in shuffleTypes:
            n = 0

            x1 = Counter()
            x2 = Counter()
            x3 = Counter()
            
            print("Processing species %d (%s), shuffleType=%d" % (taxid, getSpeciesFileName(taxid), shuffleType))

            profileId = "%d_%d_%s_t%d" % (args.profile[0], args.profile[1], args.profile[2], shuffleType)

            nativeMeanProfile = MeanProfile( profileLength(args.profile) )
            shuffledMeanProfile = MeanProfile( profileLength(args.profile) )

            shuffled25Profile = MeanProfile( profileLength(args.profile) )
            shuffled75Profile = MeanProfile( profileLength(args.profile) )

            h5destination     = PlottingDestination(args, shuffleType, taxid, wilcoxonDLFEZero=True, wilcoxonDLFE50_100=True)
            h5destination_tag = PlottingDestination(args, shuffleType, taxid, subclass="endcodon_tag")
            h5destination_taa = PlottingDestination(args, shuffleType, taxid, subclass="endcodon_taa")
            h5destination_tga = PlottingDestination(args, shuffleType, taxid, subclass="endcodon_tga")
            
            h5destination_readthrough_pos = PlottingDestination(args, shuffleType, taxid, subclass="readthrough_pos")
            h5destination_readthrough_neg = PlottingDestination(args, shuffleType, taxid, subclass="readthrough_neg")

            h5destination_next_cds_same     = PlottingDestination(args, shuffleType, taxid, subclass="next_cds_same")
            h5destination_next_cds_opposite = PlottingDestination(args, shuffleType, taxid, subclass="next_cds_opposite")

            h5destination_integenic_short    = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_positive_short")
            h5destination_integenic_long     = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_positive_long")
            h5destination_integenic_overlap  = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_positive_overlap")

            h5destination_integenic_negative = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_negative")
            h5destination_integenic_0_19     = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_0_19")
            h5destination_integenic_10_29    = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_10_29")
            h5destination_integenic_20_39    = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_20_39")
            h5destination_integenic_30_49    = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_30_49")
            h5destination_integenic_40_59    = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_40_59")
            h5destination_integenic_50_69    = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_50_69")
            h5destination_integenic_60_79    = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_60_79")
            h5destination_integenic_70_89    = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_70_89")
            h5destination_integenic_80_999   = PlottingDestination(args, shuffleType, taxid, subclass="intergenic_80_999")

            h5destination_strand_positive     = PlottingDestination(args, shuffleType, taxid, subclass="strand_positive")
            h5destination_strand_negative     = PlottingDestination(args, shuffleType, taxid, subclass="strand_negative")

            h5destination_operon_first       = PlottingDestination(args, shuffleType, taxid, subclass="operon_first")
            h5destination_operon_not_first   = PlottingDestination(args, shuffleType, taxid, subclass="operon_not_first")
            h5destination_operon_last        = PlottingDestination(args, shuffleType, taxid, subclass="operon_last")
            h5destination_operon_not_last    = PlottingDestination(args, shuffleType, taxid, subclass="operon_not_last")
            
            h5destination_operon_not_last_aug    = PlottingDestination(args, shuffleType, taxid, subclass="operon_not_last_with_aug")
            h5destination_operon_not_last_no_aug    = PlottingDestination(args, shuffleType, taxid, subclass="operon_not_last_without_aug")
            h5destination_operon_last_aug    = PlottingDestination(args, shuffleType, taxid, subclass="operon_last_with_aug")
            h5destination_operon_last_no_aug    = PlottingDestination(args, shuffleType, taxid, subclass="operon_last_without_aug")

            h5destination_operon_1st_and_not_last  = PlottingDestination(args, shuffleType, taxid, subclass="operon_1st_and_not_last")
            h5destination_operon_2nd_and_not_last  = PlottingDestination(args, shuffleType, taxid, subclass="operon_2nd_and_not_last")
            h5destination_operon_3rd_and_not_last  = PlottingDestination(args, shuffleType, taxid, subclass="operon_3rd_and_not_last")
            h5destination_operon_last_and_not_first  = PlottingDestination(args, shuffleType, taxid, subclass="operon_last_and_not_first")
            h5destination_operon_2nd_last  = PlottingDestination(args, shuffleType, taxid, subclass="operon_2nd_last")
            h5destination_operon_3rd_last  = PlottingDestination(args, shuffleType, taxid, subclass="operon_3rd_last")


            h5destination_operon_last_3utr_0  = PlottingDestination(args, shuffleType, taxid, subclass="operon_last_3utr_0")
            h5destination_operon_last_3utr_1_49 = PlottingDestination(args, shuffleType, taxid, subclass="operon_last_3utr_1_49")
            h5destination_operon_last_3utr_0_49 = PlottingDestination(args, shuffleType, taxid, subclass="operon_last_3utr_0_49")
            h5destination_operon_last_3utr_50 = PlottingDestination(args, shuffleType, taxid, subclass="operon_last_3utr_50")

            h5destination_pa_low          = PlottingDestination(args, shuffleType, taxid, subclass="pa_low")
            h5destination_pa_med          = PlottingDestination(args, shuffleType, taxid, subclass="pa_med")
            h5destination_pa_high         = PlottingDestination(args, shuffleType, taxid, subclass="pa_high")
            h5destination_pa_logratio_neg = PlottingDestination(args, shuffleType, taxid, subclass="pa_logratio_neg")
            h5destination_pa_logratio_0   = PlottingDestination(args, shuffleType, taxid, subclass="pa_logratio_0")
            h5destination_pa_logratio_pos = PlottingDestination(args, shuffleType, taxid, subclass="pa_logratio_pos")
            
            dLFEvsFlankingLength = pd.DataFrame({'flanking_length':pd.Series(dtype='int'), 'dLFE':pd.Series(dtype='float'), 'nextCDSOppositeStrand':pd.Series(dtype='int')})

            PAstats = pd.DataFrame({'PA':pd.Series(dtype='float'), 'mean_dLFE_for_peak':pd.Series(dtype='float'), '3utr_length':pd.Series(dtype='int')})
            PAtrends = pd.DataFrame({'upstream_PA':pd.Series(dtype='float'), 'downstream_PA':pd.Series(dtype='float'), 'mean_dLFE_for_peak':pd.Series(dtype='float'), '3utr_length':pd.Series(dtype='int')})

            

            #dfOut = pd.DataFrame({'ProtId':pd.Series(dtype='str'), 'GeneName':pd.Series(dtype='str'), 'IntergenicLengthNt':pd.Series(dtype='int')})

           
            xRange = profileElements(args.profile)

            # nativeMeanProfile_HighPAOnly = None
            # nativeMeanProfile_MediumPAOnly = None
            # nativeMeanProfile_LowPAOnly = None
            # if( args.pax_db ):
            #     nativeMeanProfile_HighPAOnly = MeanProfile( profileLength(args.profile) )
            #     nativeMeanProfile_MediumPAOnly = MeanProfile( profileLength(args.profile) )
            #     nativeMeanProfile_LowPAOnly = MeanProfile( profileLength(args.profile) )

            GCProfile = MeanProfile( profileLength(args.profile) )

            nextStopCodonPositions = []

            #deltasForWilcoxon = np.zeros((0,2), dtype=float)
            deltasForWilcoxon = pd.DataFrame({'pos':pd.Series(dtype='int'), 'delta':pd.Series(dtype='float')})

            dLFEMatrix  = np.zeros((0,200))
            dLFEMatrix2 = np.zeros((0,200))

            fullDeltas = []
            hist2dProfileData = np.array(())
            hist2dProfilePos  = np.array(())

            geneLevelScatter = pd.DataFrame({'gc':pd.Series(dtype='float'), 'logpval':pd.Series(dtype='float'), 'abslogpval':pd.Series(dtype='float'), 'protid':pd.Series(dtype='str')})

            cdsLengths = []

            fullSeqs = []
            dfCodonw = None

            gm = getGenomeModelFromCache( taxid )

            if args.pa:
                proteinAbundance    = getSpeciesPaxdbData( taxid, convertToPercentiles=False )
                proteinAbundanceRel = getSpeciesPaxdbData( taxid, convertToPercentiles=True  )

            print("-----"*10)
            print("Reading items...")
            print("-----"*10)
            allCDSrecords = list( sampleProfilesFixedIntervals(
                convertResultsToMFEProfiles(
                    readSeriesResultsForSpeciesWithSequence((args.computation_tag,), taxid, numShuffledGroups, numShuffledGroups, shuffleType=shuffleType )
                    , numShuffledGroups)
                , args.profile[3], args.profile[0], args.profile[1], args.profile[2])
                                  )
            print("-----"*10)

            allUTRlengths = []
            UTRlengthIndex = []
            PAratioIndex = []
            PAratioUTRlengthIndex = []

            # ------------------------------------
            # Process all CDS for this species
            # ------------------------------------
            for result in allCDSrecords:

                fullCDS = result["cds-seq"]

                # This is not correct (for stop3utr) but not really used...
                if args.profile[2]=="stop3utr":
                    seq = "N"*(args.profile[0])
                else:
                    seq = fullCDS[args.profile[3]:args.profile[0]]

                if not seq:
                    continue

                                    

                stopCodonPos = result['content'][0]['stop-codon-pos']
                assert(stopCodonPos%3==0)
                stopCodon = fullCDS[stopCodonPos-3:stopCodonPos]
                stopCodonFreq.update( (stopCodon, ) )

                cds = result["cds"]
                
                protId = cds.getProtId()

                feature = gm.findFeatureById( protId )
                
                flanking3UTRRegionLengthNt = cds.flankingRegion3UtrLength()

                #print("Length: {}nt".format(result["cds"].length()))

                fullSeqs.append( SeqRecord( Seq(fullCDS, NucleotideAlphabet), id=protId) )


                profileData = result["profile-data"]
                #print("ppl: {}".format( profileData.shape))
                assert(profileData.shape[0] >= numShuffledGroups)
                #print(profileData.shape)
                #print(profileData)

                #print(profileData[:,0].T)

                #print( profileData[:, [0,99,-1]] )
                #print(profileData.shape)

                # Prepare mean MFE profiles

                h5destination.accumulate( profileData, seq, fullCDS, positions={'stop-codon':stopCodonPos, '3utr-length':flanking3UTRRegionLengthNt } )
                
                if stopCodon=="tag":
                    h5destination_tag.accumulate( profileData, seq, fullCDS )
                    
                elif stopCodon=="taa":
                    h5destination_taa.accumulate( profileData, seq, fullCDS )
                    
                elif stopCodon=="tga":
                    h5destination_tga.accumulate( profileData, seq, fullCDS )


                nextStartCodonPos = fullCDS[stopCodonPos:].find('atg')
                nextStopCodonPositions.append( nextStartCodonPos )


                strand = feature[1].data['strand']
                if strand=='+':
                    h5destination_strand_positive.accumulate( profileData, seq, fullCDS )
                elif strand=='-':
                    h5destination_strand_negative.accumulate( profileData, seq, fullCDS )
                else:
                    print("Invalid strand detected for protId={}".format(protId))

                gid = cds.getGeneId()
                oinf = cds.getOperonInfo()

                for i in [gid]+gm.findEquivalentIdentifiers(gid):
                    utr3Length = utr3primeLengths.get( i, None )
                    if not utr3Length is None: break

                #if (not oinf is None) and (oinf[1] > 1): # only include operonic genes
                if (not oinf is None):

                    operonPosFromStart = oinf[0]
                    operonPosFromEnd   = oinf[1]-1 - oinf[0]
                    assert(operonPosFromStart >= 0)
                    assert(operonPosFromEnd >= 0)

                    if operonPosFromStart==0:
                        assert(oinf[0]==0)
                        h5destination_operon_first.accumulate( profileData, seq, fullCDS )
                            
                    else:
                        h5destination_operon_not_first.accumulate( profileData, seq, fullCDS )
                        
                    if operonPosFromEnd==0:
                        assert( oinf[0]==oinf[1]-1 )
                        h5destination_operon_last.accumulate( profileData, seq, fullCDS )

                        if operonPosFromStart != 0:
                            h5destination_operon_last_and_not_first.accumulate( profileData, seq, fullCDS )

                        if not utr3Length is None:

                            if utr3Length == 0:
                                h5destination_operon_last_3utr_0.accumulate( profileData, seq, fullCDS )
                            elif utr3Length <= 49:
                                h5destination_operon_last_3utr_1_49.accumulate( profileData, seq, fullCDS )
                            
                            if utr3Length <= 49:
                                h5destination_operon_last_3utr_0_49.accumulate( profileData, seq, fullCDS )
                            else:
                                h5destination_operon_last_3utr_50.accumulate( profileData, seq, fullCDS )
                        #################################################################################
                        # posArray = np.repeat(np.expand_dims(np.array(xRange), axis=0), 20, axis=0)
                        # deltasArray = profileData[0,None] - profileData[1:]
                        # #assert( posArray.shape == deltasArray.shape )
                        # if ( posArray.shape == deltasArray.shape ):
                        #     # remove NaNs
                        #     validIndices = ~np.isnan( deltasArray )
                        #     posArray = posArray[validIndices]
                        #     deltasArray = deltasArray[validIndices]
                        #     # store the values
                        #     hist2dProfileData = np.append( hist2dProfileData, deltasArray.flat )
                        #     hist2dProfilePos  = np.append( hist2dProfilePos,  posArray.flat )
                        #################################################################################
                        

                        if nextStartCodonPos < 0 or nextStartCodonPos > 50: # no nearby start codon
                            h5destination_operon_last_no_aug.accumulate( profileData, seq, fullCDS )
                        else:
                            h5destination_operon_last_aug.accumulate( profileData, seq, fullCDS )
                            
                    else: # not last
                        h5destination_operon_not_last.accumulate( profileData, seq, fullCDS )

                        if nextStartCodonPos < 0 or nextStartCodonPos > 50: # no nearby start codon
                            h5destination_operon_not_last_no_aug.accumulate( profileData, seq, fullCDS )
                        else:
                            h5destination_operon_not_last_aug.accumulate( profileData, seq, fullCDS )

                        if operonPosFromStart == 0:
                            h5destination_operon_1st_and_not_last.accumulate( profileData, seq, fullCDS )
                        elif operonPosFromStart == 1:
                            h5destination_operon_2nd_and_not_last.accumulate( profileData, seq, fullCDS )
                        elif operonPosFromStart == 2:
                            h5destination_operon_3rd_and_not_last.accumulate( profileData, seq, fullCDS )
                        
                else:
                    operonPosFromStart = None
                    operonPosFromEnd   = None
                            
                            

                # #################################################################################
                posArray = np.repeat(np.expand_dims(np.array(xRange), axis=0), 20, axis=0)
                deltasArray = profileData[0,None] - profileData[1:]
                #assert( posArray.shape == deltasArray.shape )
                if ( posArray.shape == deltasArray.shape ):
                    # remove NaNs
                    validIndices = ~np.isnan( deltasArray )
                    posArray = posArray[validIndices]
                    deltasArray = deltasArray[validIndices]
                    # store the values
                    hist2dProfileData = np.append( hist2dProfileData, deltasArray.flat )
                    hist2dProfilePos  = np.append( hist2dProfilePos,  posArray.flat )

                    if args.dump:
                        frdump.write("{},{},{},{}\n".format( cds.getGeneId(), cds.getProtId(), flanking3UTRRegionLengthNt, ",".join(map(lambda x: "{}".format(round(x,3)), np.mean( profileData[0,None] - profileData[1:], axis=0)))))
                # #################################################################################
                       

                # posArray = np.repeat(np.expand_dims(np.array(xRange), axis=0), 20, axis=0)
                # deltasArray = profileData[0,None] - profileData[1:]
                # #assert( posArray.shape == deltasArray.shape )
                # if ( posArray.shape == deltasArray.shape ):
                #     # remove NaNs
                #     validIndices = ~np.isnan( deltasArray )
                #     posArray = posArray[validIndices]
                #     deltasArray = deltasArray[validIndices]
                #     # store the values
                #     hist2dProfileData = np.append( hist2dProfileData, deltasArray.flat )
                #     hist2dProfilePos  = np.append( hist2dProfilePos,  posArray.flat )

                #
                nextCDSOnOppositeStrand = cds.nextCDSOnOppositeStrand()
                assert(isinstance(nextCDSOnOppositeStrand, bool))
                if nextCDSOnOppositeStrand:
                    h5destination_next_cds_opposite.accumulate( profileData, seq, fullCDS )
                else:
                    h5destination_next_cds_same.accumulate( profileData, seq, fullCDS )
                    

                if profileData.shape[1] > 105:
                    dLFE_105 = profileData[0,105] - np.mean(profileData[1:,105]) 
                    dLFEvsFlankingLength = dLFEvsFlankingLength.append(
                        pd.DataFrame({'flanking_length':pd.Series([flanking3UTRRegionLengthNt]),
                                      'dLFE':pd.Series([dLFE_105]),
                                      'nextCDSOppositeStrand':pd.Series([int(nextCDSOnOppositeStrand)])}) )
                else:
                    print("Warning: skipping {}...".format( protId ))

                if flanking3UTRRegionLengthNt > 0     and flanking3UTRRegionLengthNt <= 100:
                    h5destination_integenic_short.accumulate( profileData, seq, fullCDS )
                elif flanking3UTRRegionLengthNt > 100 and flanking3UTRRegionLengthNt <= 400:
                    h5destination_integenic_long.accumulate( profileData, seq, fullCDS )
                elif flanking3UTRRegionLengthNt <= 0:
                    h5destination_integenic_overlap.accumulate( profileData, seq, fullCDS )

                #if 'gene-name' in feature[1].data:
                #    dfOut = dfOut.append( pd.DataFrame({'ProtId':pd.Series([protId]), 'GeneName':pd.Series([feature[1].data['gene-name'][0]]), 'IntergenicLengthNt':pd.Series([flanking3UTRRegionLengthNt])}) )

                cs = []
                if flanking3UTRRegionLengthNt >= 0     and flanking3UTRRegionLengthNt <= 19:
                    cs.append( h5destination_integenic_0_19 )
                if flanking3UTRRegionLengthNt >= 10  and flanking3UTRRegionLengthNt <= 29:
                    cs.append( h5destination_integenic_10_29 )
                if flanking3UTRRegionLengthNt >= 20  and flanking3UTRRegionLengthNt <= 39:
                    cs.append( h5destination_integenic_20_39 )
                if flanking3UTRRegionLengthNt >= 30  and flanking3UTRRegionLengthNt <= 49:
                    cs.append( h5destination_integenic_30_49 )
                if flanking3UTRRegionLengthNt >= 40  and flanking3UTRRegionLengthNt <= 59:
                    cs.append( h5destination_integenic_40_59 )
                if flanking3UTRRegionLengthNt >= 50  and flanking3UTRRegionLengthNt <= 69:
                    cs.append( h5destination_integenic_50_69 )
                if flanking3UTRRegionLengthNt >= 60  and flanking3UTRRegionLengthNt <= 79:
                    cs.append( h5destination_integenic_60_79 )
                if flanking3UTRRegionLengthNt >= 70  and flanking3UTRRegionLengthNt <= 89:
                    cs.append( h5destination_integenic_70_89 )
                if flanking3UTRRegionLengthNt >= 80:
                    cs.append( h5destination_integenic_80_999 )
                if flanking3UTRRegionLengthNt < 0:
                    cs.append( h5destination_integenic_negative )
                for c in cs:
                    c.accumulate( profileData, seq, fullCDS )
                del(cs)

                allUTRlengths.append( flanking3UTRRegionLengthNt )



                if args.pa:
                    # Find PA for this protein
                    PA = proteinAbundance.get( gid, None )
                    assert( (PA is None) or (PA >= 0.0) )
                    PArel = proteinAbundanceRel.get( gid, None )
                    assert( (PA is None) == (PArel is None) )
                    
                    if not PA is None:

                        if ( profileData.shape[1] == 200 ): # TODO FIX THIS
                            meanOfPeak = np.mean(np.mean(profileData[0,None] - profileData[1:], axis=0)[95:110])
                        else:
                            meanOfPeak = None
                            
                        # Store data in 1-protein data-set
                        PAstats = PAstats.append( pd.DataFrame({'PA':pd.Series([PA]), 'mean_dLFE_for_peak':pd.Series([meanOfPeak]), '3utr_length':pd.Series([flanking3UTRRegionLengthNt])}) )
                        if PArel   > 0.7:
                            h5destination_pa_high.accumulate( profileData, seq, fullCDS )
                        elif PArel >= 0.3:
                            h5destination_pa_med.accumulate(  profileData, seq, fullCDS )
                        else:
                            assert(PArel>=0.0 and PArel < 0.3)
                            h5destination_pa_low.accumulate(  profileData, seq, fullCDS )
                            

                        # Try to find PA data for downstream protein
                        #------------------------------------------------

                        downstreamFeature = gm.moleculeModels[ feature[0] ].find3PrimeFlankingRegion( feature[1] )['downstream-feature']                     # Find gid for downstream feature
                        downstreamFeatureGid = downstreamFeature.data['gene-id']
                        if downstreamFeatureGid[:5]=="gene:":
                            downstreamFeatureGid = downstreamFeatureGid[5:]  # remove the 'gene:' prefix

                        downstreamFeaturePA = proteinAbundance.get( downstreamFeatureGid, None )
                        assert( (downstreamFeaturePA is None) or (downstreamFeaturePA >= 0.0) )

                        if (not downstreamFeaturePA is None) and (not nextCDSOnOppositeStrand):

                            #PAtrends = PAtrends.append( pd.DataFrame({'upstream_PA':pd.Series([PA]), 'downstream_PA':pd.Series([downstreamFeaturePA]), 'mean_dLFE_for_peak':pd.Series([meanOfPeak]), '3utr_length':pd.Series([flanking3UTRRegionLengthNt]) }) )


                            if PA > 1e-10:
                                if downstreamFeaturePA > 1e-10:
                                    logPAratio = log10( downstreamFeaturePA / PA )
                                else:
                                    logPAratio = -10.0 # TODO FIX THIS
                            else:
                                if downstreamFeaturePA > 1e-10:
                                    logPAratio = 10.0
                                else:
                                    logPAratio = None

                            # Store data for smoothed plot:
                            if not logPAratio is None:
                                newDeltas = profileData[0,0::1] - np.mean(profileData[1:,0::1], axis=0)
                                if newDeltas.shape[0] == 200: # TODO FIX THIS

                                    #if logPAratio <= -1.0: # neg
                                    #if logPAratio > -1.0 and logPAratio < 1.0: # 0
                                    if logPAratio >= 1.0: # pos
                                        dLFEMatrix2 = np.append( dLFEMatrix2, np.expand_dims( newDeltas, axis=0 ), axis=0 )
                                        PAratioIndex.append( logPAratio )
                                        PAratioUTRlengthIndex.append( flanking3UTRRegionLengthNt )

                                if logPAratio > 1.0:
                                    h5destination_pa_logratio_pos.accumulate( profileData, seq, fullCDS )
                                elif logPAratio >= -1.0:
                                    h5destination_pa_logratio_0.accumulate( profileData, seq, fullCDS )
                                else:
                                    assert( logPAratio < -1.0 )
                                    h5destination_pa_logratio_neg.accumulate( profileData, seq, fullCDS )

                                #if not operonPosFromStart is None:
                                #    #print("{} {}".format(operonPosFromStart, operonPosFromEnd))
                                #    #if operonPosFromEnd > 0:
                                #    #    PAtrends = PAtrends.append( pd.DataFrame({'upstream_PA':pd.Series([PA]), 'downstream_PA':pd.Series([downstreamFeaturePA]), 'mean_dLFE_for_peak':pd.Series([meanOfPeak]), '3utr_length':pd.Series([flanking3UTRRegionLengthNt]) }) )
                                PAtrends = PAtrends.append( pd.DataFrame({'upstream_PA':pd.Series([PA]), 'downstream_PA':pd.Series([downstreamFeaturePA]), 'mean_dLFE_for_peak':pd.Series([meanOfPeak]), '3utr_length':pd.Series([flanking3UTRRegionLengthNt]) }) )
                                    
                                    
                            

                
                
                exid = 0
                experiment_name = ("ribo_MG1655_MOPS_rep1", "ribo_MG1655_MOPS_rep2", "ribo_rich", "WT_rep1", "WT_rep2", "WT_rep3")[exid]
                readthroughVal = getCDSProperty( taxid, protId, "readthrough-v2.ex{}".format(exid) )
                #print("readthrough-v2.ex0:{}".format(readthroughVal))
                if readthroughVal=="1":
                    h5destination_readthrough_pos.accumulate( profileData, seq, fullCDS )
                    
                elif readthroughVal=="0":
                    h5destination_readthrough_neg.accumulate( profileData, seq, fullCDS )

                # Prepare data for genome-wide wilcoxon test
                #newDeltas = profileData[0,0::4] - np.mean(profileData[1:,0::4], axis=0)
                newDeltas = profileData[0,0::1] - np.mean(profileData[1:,0::1], axis=0)
                #print("newDeltas: {}".format(newDeltas.shape))
                #newPositions = range(args.profile[3], profileLength(args.profile), 40)
                newPositions = list(range(args.profile[3], args.profile[0], args.profile[1]))
                deltaspd = pd.DataFrame({'pos':pd.Series(newPositions, dtype='int'), 'delta':pd.Series(newDeltas, dtype='float')})
                #print("deltaspd: {}".format(deltaspd.shape))
                deltasForWilcoxon = deltasForWilcoxon.append(deltaspd)

                if ( posArray.shape == deltasArray.shape ):
                    dLFEMatrix = np.append( dLFEMatrix, np.expand_dims( newDeltas, axis=0 ), axis=0 )
                    UTRlengthIndex.append( flanking3UTRRegionLengthNt )
                else:
                    print("Warning: skipped {}, flankingLegth={}, strand={}".format(protId, flanking3UTRRegionLengthNt, strand ))
                
                fullDeltas.append( profileData[0,0::1] - profileData[1:,0::1] )  # store the 20x31 matrix of deltas for full wilcoxon test

                # Prepare data for GC vs. selection test
                meanGC = calcSampledGCcontent( seq, 10000)[0]
                if( not (meanGC >= 0.05 and meanGC <= 0.95)):
                    meanGC = None
                #deltas = profileData[0,0::4] - np.mean(profileData[1:,0::4], axis=0)
                deltas = profileData[0,0::1] - np.mean(profileData[1:,0::1], axis=0)
                #print("deltas: {}".format(deltas.shape))
                pvalue = wilcoxon(deltas).pvalue
                direction = np.mean(deltas)
                directedLogPval = None

                if( pvalue > 0.0 ):
                    directedLogPval = log10(pvalue) * direction * -1.0
                else:
                    directedLogPval = -250.0      * direction * -1.0

                paval = None
                # if( args.pax_db ):
                #     paval = pa.get(protId)
                #     if( paval >= 0.8 ):
                #         nativeMeanProfile_HighPAOnly.add( profileData[0,None] )
                #     elif( paval <= 0.2 ):
                #         nativeMeanProfile_LowPAOnly.add( profileData[0,None] )
                #     elif( not paval is None ):
                #         nativeMeanProfile_MediumPAOnly.add( profileData[0,None] )

                cds_length_nt = len(fullCDS)
                cdsLengths.append(cds_length_nt)



                frline = "{}\t{}\t{}".format(protId, stopCodon, deltas)
                frout.write(frline)
                #print(frline)
                
                geneLevelScatter = geneLevelScatter.append(pd.DataFrame({'gc':pd.Series([meanGC]), 'logpval': pd.Series([directedLogPval]), 'abslogpval': pd.Series([pvalue]), 'protid':pd.Series([protId]), 'pa':pd.Series([paval]), 'cds_length_nt':pd.Series([cds_length_nt])}))


                x1.update((fullCDS[0],))
                x2.update((fullCDS[1],))
                x3.update((fullCDS[2],))
                del fullCDS

                del result
                n += 1
                
                if( rl()):
                    print("# {} - {} records done".format(datetime.now().isoformat(), n))
                
            #del(pvalue); del(direction); del(seq); del(deltas)

            if args.pa and PAtrends.shape[0]:
                PAtrends['PAratio'] = np.log(PAtrends.downstream_PA / PAtrends.upstream_PA)
                PAtrends['upstream_PA.']  = percentileRanks( PAtrends['upstream_PA'] )
                PAtrends['downstream_PA.'] = percentileRanks( PAtrends['downstream_PA'] )
                PAstats['PA.'] = percentileRanks( PAstats['PA'] )
                
                print("%%%%"*5)
                print( "PA1prot: {} PA2prot: {}".format( PAstats.shape[0], PAtrends.shape[0] ))

                scatterPlot( taxid, profileId, PAtrends, "upstream_PA.",       "downstream_PA.", label="pa2" )
                
                scatterPlot( taxid, profileId, PAstats,  "mean_dLFE_for_peak", "PA.",            label="pa_dLFE_peak" )
                
                scatterPlot( taxid, profileId, PAstats,  "3utr_length",        "PA.",            label="pa_utr_length" )

                scatterPlot( taxid, profileId, PAtrends, "mean_dLFE_for_peak", "downstream_PA.", label="downstream_pa_dLFE_peak" )
                
                scatterPlot( taxid, profileId, PAtrends, "mean_dLFE_for_peak", "PAratio",        label="pa_ratio_dLFE_peak" )


                trends_near_strong = PAtrends[(PAtrends['mean_dLFE_for_peak']< -3) & (PAtrends['3utr_length']<=25)]
                trends_near_weak   = PAtrends[(PAtrends['mean_dLFE_for_peak']>=-3) & (PAtrends['3utr_length']<=25)]
                trends_far_strong  = PAtrends[(PAtrends['mean_dLFE_for_peak']< -3) & (PAtrends['3utr_length']> 25)]
                trends_far_weak    = PAtrends[(PAtrends['mean_dLFE_for_peak']>=-3) & (PAtrends['3utr_length']> 25)]

                PAtrends['IsNear'] = False
                PAtrends.loc[ PAtrends['3utr_length']<=25, 'IsNear' ] = True
                #trends_near = PAtrends[ PAtrends['3utr_length']<=25 ]
                #trends_far  = PAtrends[ PAtrends['3utr_length']> 25 ]
                
                ratios_near_strong = trends_near_strong.PAratio.values
                ratios_near_weak   = trends_near_weak.PAratio.values
                ratios_far_strong  = trends_far_strong.PAratio.values
                ratios_far_weak    = trends_far_weak.PAratio.values

                #wilcoxon( trends_far_weak.PAratio )
                #trends_far_weak.PAratio.median()
                #trends_far_weak.PAratio.mean()

                # ratios_near_strong = PAtrends[(PAtrends['mean_dLFE_for_peak']< -3) & (PAtrends['3utr_length'].between(0,24.9999))].PAratio.values
                # ratios_near_weak   = PAtrends[(PAtrends['mean_dLFE_for_peak']>=-3) & (PAtrends['3utr_length'].between(0,24.9999))].PAratio.values
                # ratios_far_strong  = PAtrends[(PAtrends['mean_dLFE_for_peak']< -3) & (PAtrends['3utr_length'].between(25,50))].PAratio.values
                # ratios_far_weak    = PAtrends[(PAtrends['mean_dLFE_for_peak']>=-3) & (PAtrends['3utr_length'].between(25,50))].PAratio.values

                labels = []
                dfs = (trends_near_strong, trends_near_weak, trends_far_strong, trends_far_weak)
                for df, tag in zip(dfs, ('near_strong', 'near_weak', 'far_strong', 'far_weak')):
                    labels.append( "{} (p-val: {:.2} median: {:.2})".format( tag, wilcoxon( df.PAratio ).pvalue, df.PAratio.median() ) )
                    
                    
                
                # red    - near_strong
                # blue   - near_weak
                # yellow - far_strong
                # black  - far_weak
                plotHistogramComparison( taxid, (ratios_near_strong, ratios_near_weak, ratios_far_strong, ratios_far_weak), labels )

                plotPAregression( taxid, PAtrends )

                plotPAratioVsDistanceHeatmap( taxid, PAtrends )
                                

                
            # Refuse to proceed if the data found is unreasonably small
            if( n < 100 ):
                raise Exception("Found insufficient data to process taxid=%d (n=%d)" % (taxid, n))

            if args.plot_adaptive_lengths:
                numGroups = 10
                positiveLengths = [x for x in sorted(allUTRlengths) if x>=0]
                targetSize = len(positiveLengths) // (numGroups-1) # 2 special groups - negative and 'last' group
                groups = []
                #currentMax = 0
                for candidate in sorted(frozenset(positiveLengths)):
                    #print("candidate:")
                    currentSize = len([x for x in positiveLengths if x<=candidate])
                    #print("{}) Candidate: {} size: {}".format(len(groups), candidate, currentSize))
                    
                    if currentSize > targetSize: 
                        groups.append(candidate) 
                        positiveLengths = [x for x in positiveLengths if x>candidate] 
                groups.append(max(allUTRlengths))  # The last group contains all items left
                #assert(len(groups)==numGroups-1)

                # Create dLFE destinations for all groups
                h5destination_integenic_adaptive_neg  = PlottingDestination(args, shuffleType, taxid, subclass="intergenic.a_neg")
                allAdaptiveDestinations = [PlottingDestination(args, shuffleType, taxid, subclass="intergenic.a_p{}".format(i)) for i in range(1,len(groups)+1)]
                actualGroupSizes = dict(zip(range(10), [0]*10))
                
                for result in allCDSrecords:
                    cds = result["cds"]
                    protId = cds.getProtId()
                    fullCDS = result["cds-seq"]
                    profileData = result["profile-data"]
                    
                    if args.profile[2]=="stop3utr":
                        seq = "N"*(args.profile[0])
                    else:
                        seq = fullCDS[args.profile[3]:args.profile[0]]
                        
                    
                    flanking3UTRRegionLengthNt = cds.flankingRegion3UtrLength()

                    if flanking3UTRRegionLengthNt<0:
                        h5destination_integenic_adaptive_neg.accumulate( profileData, seq, fullCDS )
                        actualGroupSizes[0] += 1
                    else:
                        matchingGroup = next( n for n,i in enumerate([-1]+groups) if i>=flanking3UTRRegionLengthNt ) # Find group to hold item; return 1-based group index
                        assert(matchingGroup>=1)
                        assert(matchingGroup<=len(groups))
                        allAdaptiveDestinations[matchingGroup-1].accumulate( profileData, seq, fullCDS )
                        actualGroupSizes[matchingGroup] += 1
                        
                    if( rl()):
                        print("# {} - {} records done".format(datetime.now().isoformat(), n))
                print("====="*10)
                print(actualGroupSizes)

                plotMultipleDestinations( taxid, "utr3prime_adaptive", [h5destination_integenic_adaptive_neg] + allAdaptiveDestinations, [(-30,-1)] + [(x+1,y) for x,y in zip([-1]+groups, groups)], xRange )


            if args.utr3prime_length_effect_plot:
                # (1/20.0*sqrt(2*pi))*exp(-((x-5)**2)/20.0)
                sigma = 100.0
                
                
                d = dLFEMatrix.copy()
                d[np.isnan(d)] = 0  # TODO is this ok?

                smoothedLengthEffect = getSmoothedPropertyEffectOnProfiles( d, UTRlengthIndex, sigma=sigma )
                #plotSmoothedLengthEffectData( taxid, 'smoothed', smoothedLengthEffect )

                bsdata = pd.DataFrame( {'Pos':pd.Series(dtype='int'), 'MinDLFE':pd.Series(dtype='float')} )
                N = len(UTRlengthIndex)

                for i in range(10):
                    print("sample {}...".format(i))
                    sample = random.choices( range(N), k=N )
                    # resample the data
                    d2 = d[sample,:]
                    
                    # resample the property values
                    propValues = list(map( lambda i: UTRlengthIndex[i], sample ))
                    
                    # calculate the smoothed curve
                    bsdata = bsdata.append( getSmoothedPropertyEffectOnProfiles( d2, propValues, sigma=sigma ))

                smoothedLengthEffectWithStats = pd.DataFrame( {'Pos':pd.Series(dtype='int'), 'MinDLFE':pd.Series(dtype='float'), 'Mean':pd.Series(dtype='float'), 'P25':pd.Series(dtype='float'), 'P75':pd.Series(dtype='float'), 'Density':pd.Series(dtype='float')} )

                for l in range(-50,300,1):
                    valuesAtPos = bsdata[bsdata['Pos']==l]["MinDLFE"]
                    populationValue = float(smoothedLengthEffect[smoothedLengthEffect['Pos']==l]['MinDLFE'])
                    density = float(smoothedLengthEffect[smoothedLengthEffect['Pos']==l]['Density'])
                    meanValue = np.mean(valuesAtPos)
                    percentile75 = np.percentile( valuesAtPos, 75 )
                    percentile25 = np.percentile( valuesAtPos, 25 )
                    
                    smoothedLengthEffectWithStats = smoothedLengthEffectWithStats.append( pd.DataFrame( {'Pos':pd.Series([l]), 'MinDLFE':pd.Series([populationValue]), 'Mean':pd.Series([meanValue]), 'P25':pd.Series([percentile25]), 'P75':pd.Series([percentile75]), 'Density':pd.Series([density])} ))
                    
                plotSmoothedLengthEffectData( taxid, 'smoothed', smoothedLengthEffectWithStats )


            if args.pa:

                plotMultipleDestinations( taxid, "pa", (h5destination_pa_low,h5destination_pa_med,h5destination_pa_high), ((0.0,0.3),(0.3,0.7),(0.7,1.0)), xRange )
                plotMultipleDestinations( taxid, "pa_logratio", (h5destination_pa_logratio_neg, h5destination_pa_logratio_0, h5destination_pa_logratio_pos), ((-2.5,-1),(-1,1),(1,2.5)), xRange )

                
                # Smoothed PA interaction plot
                # (1/20.0*sqrt(2*pi))*exp(-((x-5)**2)/20.0)
                #sigma = 0.5
                sigma = 50
                
                #dLFEMatrix2 = np.append( dLFEMatrix2, np.expand_dims( newDeltas, axis=0 ), axis=0 )
                #PAratioIndex.append( logPAratio )

                
                d = dLFEMatrix2.copy()
                d[np.isnan(d)] = 0  # TODO is this ok?

                #smoothedLengthEffect = getSmoothedPropertyEffectOnProfiles( d, PAratioIndex, sigma=sigma, valuesRange=np.linspace(-10,10,201), valuesPropName='logPAratio', valuesPropType='float' )
                smoothedLengthEffect = getSmoothedPropertyEffectOnProfiles( d, PAratioUTRlengthIndex, sigma=sigma, valuesRange=range(-50,300,1), valuesPropName='Pos', valuesPropType='int' )
                
                #plotSmoothedLengthEffectData( taxid, 'smoothed', smoothedLengthEffect )

                #bsdata = pd.DataFrame( {'logPAratio':pd.Series(dtype='float'), 'MinDLFE':pd.Series(dtype='float'), 'MaxDLFE':pd.Series(dtype='float') } )
                bsdata = pd.DataFrame( {'Pos':pd.Series(dtype='int'), 'MinDLFE':pd.Series(dtype='float'), 'MaxDLFE':pd.Series(dtype='float') } )
                N = len(PAratioIndex)

                for i in range(10):
                    print("sample {}...".format(i))
                    sample = random.choices( range(N), k=N )
                    # resample the data
                    d2 = d[sample,:]
                    
                    # resample the property values
                    #propValues = list(map( lambda i: PAratioIndex[i], sample ))
                    propValues = list(map( lambda i: PAratioUTRlengthIndex[i], sample ))
                    
                    # calculate the smoothed curve
                    #bsdata = bsdata.append( getSmoothedPropertyEffectOnProfiles( d2, propValues, sigma=sigma, valuesRange=np.linspace(-10,10,201), valuesPropName='logPAratio', valuesPropType='float' ) )
                    bsdata = bsdata.append( getSmoothedPropertyEffectOnProfiles( d2, propValues, sigma=sigma, valuesRange=range(-50,300,1), valuesPropName='Pos', valuesPropType='int' ) )

                #smoothedLengthEffectWithStats = pd.DataFrame( {'Pos':pd.Series(dtype='int'), 'MinDLFE':pd.Series(dtype='float'), 'MaxDLFE':pd.Series(dtype='float'), 'Mean':pd.Series(dtype='float'), 'P25':pd.Series(dtype='float'), 'P75':pd.Series(dtype='float'), 'Density':pd.Series(dtype='float')} )
                smoothedLengthEffectWithStats = pd.DataFrame( {'Pos':pd.Series(dtype='int'), 'MinDLFE':pd.Series(dtype='float'), 'MaxDLFE':pd.Series(dtype='float'), 'Mean':pd.Series(dtype='float'), 'P25':pd.Series(dtype='float'), 'P75':pd.Series(dtype='float'), 'Density':pd.Series(dtype='float')} )

                #for l in np.linspace(-10,10,201):
                for l in range(-50,300,1):
                    #valuesAtPos = bsdata[np.abs(bsdata['logPAratio']-l)<0.05]["MinDLFE"]
                    valuesAtPos = bsdata[np.abs(bsdata['Pos']-l)<0.05]["MinDLFE"]
                    populationValue = float(smoothedLengthEffect[np.abs(smoothedLengthEffect['Pos']-l)<0.05]['MinDLFE'])
                    density = float(smoothedLengthEffect[np.abs(smoothedLengthEffect['Pos']-l)<0.05]['Density'])
                    meanValue = np.mean(valuesAtPos)
                    percentile75 = np.percentile( valuesAtPos, 75 )
                    percentile25 = np.percentile( valuesAtPos, 25 )
                    
                    smoothedLengthEffectWithStats = smoothedLengthEffectWithStats.append( pd.DataFrame( {'Pos':pd.Series([l]), 'MinDLFE':pd.Series([populationValue]), 'Mean':pd.Series([meanValue]), 'P25':pd.Series([percentile25]), 'P75':pd.Series([percentile75]), 'Density':pd.Series([density])} ))
                    
                #plotSmoothedLengthEffectData( taxid, 'smoothed_pa_ratio', smoothedLengthEffectWithStats, xvar='logPAratio' )
                plotSmoothedLengthEffectData( taxid, 'smoothed_pa_ratio_pos', smoothedLengthEffectWithStats, xvar='Pos' )
                

            CUBmetricsProfile = None
            if( args.codonw ):
                fFullSeqs = NamedTemporaryFile(mode="w")         # create a temporary file
                SeqIO.write( fullSeqs, fFullSeqs.name, "fasta")  # write the full sequences into the file
                dfCodonw = readCodonw( fFullSeqs.name )          # run codonw and get the gene-level results

                print('****************************************************')
                print(dfCodonw.columns)
                print(dfCodonw.head())

                print(geneLevelScatter.columns)
                print(geneLevelScatter.head())

                geneLevelScatter = pd.merge(dfCodonw, geneLevelScatter, left_index=True, right_index=False, right_on='protid')
                print(geneLevelScatter.corr())

                #args.profile[3], args.profile[0], args.profile[1]
                CUBmetricsProfile = meanCodonwProfile(fullSeqs, self.windowWidth, 'begin', args.profile[3], args.profile[0], args.profile[1]) # TODO - use real values!
                print(CUBmetricsProfile)

            #else:
            #    geneLevelScatter['CAI'] = pd.Series( np.zeros(len(geneLevelScatter)), index=geneLevelScatter.index)



            # ------------------------------------
            # Display summary for this species
            # ------------------------------------
            #print("Native:")
            #print(nativeMeanProfile.value())
            #print(nativeMeanProfile.counts())

            #print("Shuffled:")
            #print(shuffledMeanProfile.value())
            #print(shuffledMeanProfile.counts())

            #print(deltasForWilcoxon.shape)

            #------------------------------------------------------------------------------------------------------------------
            # Test for significance of the mean dLFE (postive or negative) at each position along the genome
            # (This will answer questions such as "how many genomes have (significantly) negative dLFE at position X?")
            #------------------------------------------------------------------------------------------------------------------
            # wilcoxonDf = pd.DataFrame({'pos':pd.Series(dtype='int'), 'logpval':pd.Series(dtype='float'), 'N':pd.Series(dtype='int') })
            # if( True ):
            #     print("Processing full deltas...")

            #     # Perform statistical tests based on the deltas for each position (from all proteins)
            #     for pos in range(profileLength(args.profile)):

            #         # Collect all deltas for this position (data will be an list of arrays of length 20 - one for each protein long enough to have deltas at this position)
            #         print("pos: {} len: {}".format( pos, len(fullDeltas)))
            #         data = [x[:,pos] for x in fullDeltas if x.shape[1]>pos]
            #         dataar = np.concatenate(data)  # flatten all deltas
            #         assert(dataar.ndim == 1)

            #         # Perform 1-sample Wilcoxon signed-rank test on the deltas (testing whether the deltas are symmetrical)
            #         wilcoxonPval = wilcoxon(dataar).pvalue  # 2-sided test
            #         if wilcoxonPval>0.0:
            #             logWilcoxonPval = log10(wilcoxonPval)
            #         else:
            #             logWilcoxonPval = -324.0 # ~minimum value for log10(0.000.....)

            #         N = dataar.shape[0]
                    
            #         wilcoxonDf = wilcoxonDf.append(pd.DataFrame({'pos':pd.Series(xRange[pos]), 'N':pd.Series([N]), 'logpval': pd.Series([logWilcoxonPval]) } ) )
                    
            #         #alldeltas = np.concatenate(fullDeltas)
            #     #print(wilcoxonDf)
            #     del(data); del(dataar)

            #------------------------------------------------------------------------------------------------------------------

            #------------------------------------------------------------------------------------------------------------------
            # Find "transition peak"
            #------------------------------------------------------------------------------------------------------------------
            # Calculate the dLFE
            # print("-TransitionPeak-TransitionPeak-TransitionPeak-TransitionPeak-")
            # meanDeltaLFE = nativeMeanProfile.value() - shuffledMeanProfile.value()
            # peakPos = np.argmin( meanDeltaLFE )
            # peakVal = meanDeltaLFE[peakPos]
            
            # guPeakDf = pd.DataFrame({'pos':pd.Series(dtype='int'), 'logpval':pd.Series(dtype='float') })
            
            # if peakVal <= 0.0:
            #     print("{} {}".format(peakPos, peakVal))
            #     if not wilcoxonDf[wilcoxonDf['pos']==peakPos*10].empty:
            #         logpval = wilcoxonDf[wilcoxonDf['pos']==peakPos*10].logpval.loc[0]
            #         print(type(logpval))
            #         #print(logpval.shape)
            #         print(logpval)

            #         if logpval < -2.0:
            #             #

            #             # Collect all deltas for this position (data will be an list of arrays of length 20 - one for each protein long enough to have deltas at this position)

            #             for otherPos in range(profileLength(args.profile)):

            #                 data1 = [x[:,peakPos] for x in fullDeltas if x.shape[1]>max(peakPos,otherPos)]
            #                 peakData = np.concatenate(data1)  # flatten all deltas
            #                 assert(peakData.ndim == 1)

            #                 data2 = [x[:,otherPos] for x in fullDeltas if x.shape[1]>max(peakPos,otherPos)]
            #                 otherData = np.concatenate(data2)  # flatten all deltas

            #                 assert(len(peakData)==len(otherData))
            #                 datax = otherData-peakData

            #                 print("/-: {} {} {}".format(peakPos, otherPos, np.mean(datax)))

            #                 #if( peakPos==otherPos ):
            #                 #    print(datax)

            #                 wilcoxonPval = None
            #                 if np.allclose(otherData, peakData):
            #                     logWilcoxonPval = 0.0
            #                 else:
            #                     # Perform 1-sample Wilcoxon signed-rank test on the deltas (testing whether the deltas are symmetrical)
            #                     wilcoxonPval = wilcoxon(peakData, otherData).pvalue  # 2-sided test (not ideal in this situation...)
            #                     if wilcoxonPval>0.0:
            #                         logWilcoxonPval = log10(wilcoxonPval)
            #                     elif wilcoxonPval==0.0:
            #                         logWilcoxonPval = -324.0 # ~minimum value for log10(0.000.....)
            #                     else:
            #                         logWilcoxonPval = None

            #                 if not logWilcoxonPval is None:
            #                     #guPeakDf = guPeakDf.append(pd.DataFrame({'pos':pd.Series(xRange[otherPos]), 'logpval': pd.Series([logWilcoxonPval]) } ) )


            #             print(guPeakDf)


            #------------------------------------------------------------------------------------------------------------------
            # Calculate edge-referenced wilcoxon
            #------------------------------------------------------------------------------------------------------------------

            # edgePos = profileEdgeIndex(args.profile)
            # data0 = [x[:,edgePos] if x.shape[1]>pos else None for x in fullDeltas]
            # edgeWilcoxonDf = pd.DataFrame({'pos':pd.Series(dtype='int'), 'logpval':pd.Series(dtype='float') })
            
            # for pos in range(profileLength(args.profile)):
            #     data1 = [x[:,pos] if x.shape[1]>pos else None for x in fullDeltas]
            #     assert(len(data0)==len(data1))
            #     if not data1[0] is None:
            #         print("]]]]]]]]]]]]] {}".format(data1[0].shape))

            #     diffs = []
            #     for d0, d1 in zip(data0, data1):
            #         if (not d0 is None) and (not d1 is None):
            #             #print("{} {}".format(d0.shape, d1.shape))
            #             d = d1-d0 
            #             diffs.append( d[~np.isnan(d)] )

            #     alldiffs = np.concatenate( diffs )
            #     #print(alldiffs.shape)
            #     print(pos)
            #     #print(alldiffs[:100])
            #     print(alldiffs.shape)
                
            #     wilcoxonPval = None
            #     if np.allclose(alldiffs, 0):
            #         logWilcoxonPval = 0.0
            #     else:
            #         # Perform 1-sample Wilcoxon signed-rank test on the deltas (testing whether the deltas are symmetrical)
            #         wilcoxonPval = wilcoxon(alldiffs).pvalue  # 2-sided test (not ideal in this situation...)
            #         if wilcoxonPval>0.0:
            #             logWilcoxonPval = log10(wilcoxonPval)
            #         elif wilcoxonPval==0.0:
            #             logWilcoxonPval = -324.0 # ~minimum value for log10(0.000.....)
            #         else:
            #             logWilcoxonPval = None

            #     #if not logWilcoxonPval is None:
            #         #edgeWilcoxonDf = edgeWilcoxonDf.append(pd.DataFrame({'pos':pd.Series(xRange[pos]), 'logpval': pd.Series([logWilcoxonPval]) } ))
            # #print(edgeWilcoxonDf)

            # Count the mininum number of sequences
            minCount = 1000000
            for pos in range(profileLength(args.profile)):
                countAtPos = sum([1 if x.shape[1]>pos else 0 for x in fullDeltas])
                if countAtPos < minCount: minCount = countAtPos

            #------------------------------------------------------------------------------------------------------------------
            # Store the results
            #------------------------------------------------------------------------------------------------------------------
            
            #native = np.asarray(nativeMean[1:], dtype="float")
            #shuffled = np.asarray(shuffledMean[1:], dtype="float")
            #gc = np.asarray(gcMean[1:], dtype="float")
            #xrange = [x for x in args.profile.Elements() if x<profileInfo.cdsLength()]


            #print(nativeMeanProfile.value())
            #print(xRange)
            
            #df = pd.DataFrame( { "native": nativeMeanProfile.value(), "shuffled": shuffledMeanProfile.value(), "gc":GCProfile.value(), "position": xRange, "shuffled25":shuffled25Profile.value(), "shuffled75":shuffled75Profile.value()}, index=xRange )
            #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            #if( not CUBmetricsProfile is None ):
            #    df = pd.merge( df, CUBmetricsProfile, how='left', left_on='position', right_on='windowStart')
            #    df = df.set_index('position')

            df = None
            for dest in (h5destination, h5destination_tag, h5destination_taa, h5destination_tga, h5destination_readthrough_pos, h5destination_readthrough_neg, h5destination_next_cds_same, h5destination_next_cds_opposite, h5destination_operon_not_first, h5destination_operon_first, h5destination_operon_last_no_aug, h5destination_operon_last_aug, h5destination_operon_not_last_no_aug, h5destination_operon_not_last_aug, h5destination_operon_last_3utr_0, h5destination_operon_last_3utr_1_49, h5destination_operon_last_3utr_0_49, h5destination_operon_last_3utr_50, h5destination_integenic_negative, h5destination_integenic_0_19, h5destination_integenic_10_29, h5destination_integenic_20_39, h5destination_integenic_30_49, h5destination_integenic_40_59, h5destination_integenic_50_69, h5destination_integenic_60_79, h5destination_integenic_70_89, h5destination_integenic_80_999, h5destination_operon_last_and_not_first, h5destination_operon_1st_and_not_last  ):
                dest.printDebugInfo()
                newDf = dest.finalize()
                if df is None:
                    df = newDf # only save the first df...
                
            
            combinedData[shuffleType] = df
            #print(df)

            dfProfileCorrs = None
            if( args.codonw ):
                plotMFEProfileMultiple(taxid, profileId, df, ('GC', 'Nc', 'CAI', 'CBI', 'Fop'), scaleBar=self.windowWidth)

                smfe = df['native'] - df['shuffled']
                spearman_gc  = spearmanr( df['GC'],  smfe )
                spearman_Nc  = spearmanr( df['Nc'],  smfe )
                spearman_CAI = spearmanr( df['CAI'], smfe )
                spearman_Fop = spearmanr( df['Fop'], smfe )
                dfProfileCorrs = pd.DataFrame( { "spearman_smfe_gc_rho":   spearman_gc.correlation,
                                                 "spearman_smfe_gc_pval":  spearman_gc.pvalue,
                                                 "spearman_smfe_Nc_rho":   spearman_Nc.correlation,
                                                 "spearman_smfe_Nc_pval":  spearman_Nc.pvalue,
                                                 "spearman_smfe_CAI_rho":  spearman_CAI.correlation,
                                                 "spearman_smfe_CAI_pval": spearman_CAI.pvalue,
                                                 "spearman_smfe_Fop_rho":  spearman_Fop.correlation,
                                                 "spearman_smfe_Fop_pval": spearman_Fop.pvalue },
                                                 index=(taxid,) )


            lengthsDist = np.array(cdsLengths)
            statisticsDF = pd.DataFrame({
                'mean_mean_gc': pd.Series([np.mean(GCProfile.value())]),
                'taxid': pd.Series([taxid], dtype='int'),
                'cds_count': pd.Series([len(cdsLengths)], dtype='int'),
                'media_cds_length_nt': pd.Series([np.median(cdsLengths)])
                })


            plotMFEProfileWithGC(taxid, profileId, df, computationTag=args.computation_tag)

            # plot2WayMFEComparison( taxid,
            #                        profileId,
            #                        pd.DataFrame({'native':   h5destination_readthrough_neg.getNativeProfile(),
            #                                      'shuffled': h5destination_readthrough_neg.getShuffledProfile()},
            #                                     index=xRange),
            #                        pd.DataFrame({'native':   h5destination_readthrough_pos.getNativeProfile(),
            #                                      'shuffled': h5destination_readthrough_pos.getShuffledProfile()},
            #                                     index=xRange),
            #                        computationTag=args.computation_tag,
            #                        comment=experiment_name)



            # plot2WayMFEComparison( taxid,
            #                        profileId,
            #                        pd.DataFrame({'native':   h5destination_strand_positive.getNativeProfile(),
            #                                      'shuffled': h5destination_strand_positive.getShuffledProfile()},
            #                                     index=xRange),
            #                        pd.DataFrame({'native':   h5destination_strand_negative.getNativeProfile(),
            #                                      'shuffled': h5destination_strand_negative.getShuffledProfile()},
            #                                     index=xRange),
            #                        computationTag=args.computation_tag,
            #                        yRange=(-13,1),
            #                        comment="CDS strand")

            # plot2WayMFEComparison( taxid,
            #                        profileId,
            #                        pd.DataFrame({'native':   h5destination_next_cds_same.getNativeProfile(),
            #                                      'shuffled': h5destination_next_cds_same.getShuffledProfile()},
            #                                     index=xRange),
            #                        pd.DataFrame({'native':   h5destination_next_cds_opposite.getNativeProfile(),
            #                                      'shuffled': h5destination_next_cds_opposite.getShuffledProfile()},
            #                                     index=xRange),
            #                        computationTag=args.computation_tag,
            #                        yRange=(-13,1),
            #                        comment="Next CDS on same strand")

            plotMultipleDestinations( taxid, "utr3prime_length", (h5destination_integenic_negative, h5destination_integenic_0_19, h5destination_integenic_10_29, h5destination_integenic_20_39, h5destination_integenic_30_49, h5destination_integenic_40_59, h5destination_integenic_50_69, h5destination_integenic_60_79, h5destination_integenic_70_89, h5destination_integenic_80_999), ((-30,0),(0,19),(10,29),(20,39),(30,49),(40,59),(50,69),(60,79),(70,89),(90,199)), xRange )

            (hist2d, hist2d_xedges, hist2d_yedges) = np.histogram2d( hist2dProfilePos, hist2dProfileData, bins=(199, 100), range=[[-100,100],[-30,30]], density=True )
            plot2dProfile( hist2d, hist2d_xedges, hist2d_yedges, taxid )

            plot2dProfileAsRidgeplot( hist2d, hist2d_xedges, hist2d_yedges, taxid )

            numGenesNextCDSOnSameStrand     = np.min(h5destination_next_cds_same.nativeMeanProfile.counts())
            numGenesNextCDSOnOppositeStrand  = np.min(h5destination_next_cds_opposite.nativeMeanProfile.counts())
            plotMultipleMFEComparison( taxid,
                                   profileId,
                                       (
                                        (pd.DataFrame({'native':   h5destination_next_cds_same.getNativeProfile(),
                                                       'shuffled': h5destination_next_cds_same.getShuffledProfile()},
                                                      index=xRange),
                                         "Same strand"),
                                           (pd.DataFrame({'native':   h5destination_next_cds_opposite.getNativeProfile(),
                                                          'shuffled': h5destination_next_cds_opposite.getShuffledProfile()},
                                                         index=xRange),
                                            "Opposite strand")
                                       ),
                                   computationTag=args.computation_tag,
                                   yRange=(-13,1),
                                   comment="Next CDS strand\nSame strand (N>={}) Opposite strand (N>={}) Total (N>={}) ".format( numGenesNextCDSOnSameStrand, numGenesNextCDSOnOppositeStrand, numGenesNextCDSOnSameStrand+numGenesNextCDSOnOppositeStrand  ),
                                   plotname="comparison_next_cds_strand")

            numGenesNextOperonLastUtr0     = np.min(h5destination_operon_last_3utr_0.nativeMeanProfile.counts())
            numGenesNextOperonLastUtr1_49  = np.min(h5destination_operon_last_3utr_1_49.nativeMeanProfile.counts())
            numGenesNextOperonLastUtr0_49  = np.min(h5destination_operon_last_3utr_0_49.nativeMeanProfile.counts())
            numGenesNextOperonLastUtr50    = np.min(h5destination_operon_last_3utr_50.nativeMeanProfile.counts())
            
            plotMultipleMFEComparison( taxid,
                                   profileId,
                                       (
                                        (pd.DataFrame({'native':   h5destination_operon_last_3utr_0.getNativeProfile(),
                                                       'shuffled': h5destination_operon_last_3utr_0.getShuffledProfile()},
                                                      index=xRange),
                                         "0"),
                                        (pd.DataFrame({'native':   h5destination_operon_last_3utr_1_49.getNativeProfile(),
                                                       'shuffled': h5destination_operon_last_3utr_1_49.getShuffledProfile()},
                                                      index=xRange),
                                         "1-49"),
                                        (pd.DataFrame({'native':   h5destination_operon_last_3utr_0_49.getNativeProfile(),
                                                       'shuffled': h5destination_operon_last_3utr_0_49.getShuffledProfile()},
                                                      index=xRange),
                                         "0-49"),
                                        (pd.DataFrame({'native':   h5destination_operon_last_3utr_50.getNativeProfile(),
                                                       'shuffled': h5destination_operon_last_3utr_50.getShuffledProfile()},
                                                      index=xRange),
                                         ">50")
                                       ),
                                   computationTag=args.computation_tag,
                                   yRange=(-13,1),
                                       comment="Regulondb (Lengths: 0 {}; 1-49 {}; 0-49 {}; >50 {})".format( numGenesNextOperonLastUtr0, numGenesNextOperonLastUtr1_49, numGenesNextOperonLastUtr0_49, numGenesNextOperonLastUtr50),
                                   plotname="comparison_regulondb")

            
            plotMultipleMFEComparison( taxid,
                                   profileId,
                                       (
                                        (pd.DataFrame({'native':   h5destination_operon_last_3utr_0_49.getNativeProfile(),
                                                       'shuffled': h5destination_operon_last_3utr_0_49.getShuffledProfile()},
                                                      index=xRange),
                                         "0-49"),
                                        (pd.DataFrame({'native':   h5destination_operon_last_3utr_50.getNativeProfile(),
                                                       'shuffled': h5destination_operon_last_3utr_50.getShuffledProfile()},
                                                      index=xRange),
                                         ">50")
                                       ),
                                   computationTag=args.computation_tag,
                                   yRange=(-13,1),
                                       comment="Regulondb (Lengths: 0-49 {}; >50 {})".format( numGenesNextOperonLastUtr0_49, numGenesNextOperonLastUtr50),
                                   plotname="comparison_regulondb_tidy")

            plotMultipleMFEComparison( taxid,
                                   profileId,
                                       (
                                        (pd.DataFrame({'native':   h5destination_operon_last.getNativeProfile(),
                                                       'shuffled': h5destination_operon_last.getShuffledProfile()},
                                                      index=xRange),
                                         "last"),
                                        (pd.DataFrame({'native':   h5destination_operon_1st_and_not_last.getNativeProfile(),
                                                       'shuffled': h5destination_operon_1st_and_not_last.getShuffledProfile()},
                                                      index=xRange),
                                         "1st, not last"),
                                        (pd.DataFrame({'native':   h5destination_operon_2nd_and_not_last.getNativeProfile(),
                                                       'shuffled': h5destination_operon_2nd_and_not_last.getShuffledProfile()},
                                                      index=xRange),
                                         "2nd, not last"),
                                        (pd.DataFrame({'native':   h5destination_operon_3rd_and_not_last.getNativeProfile(),
                                                       'shuffled': h5destination_operon_3rd_and_not_last.getShuffledProfile()},
                                                      index=xRange),
                                         "3rd, not last")
                                       ),
                                       computationTag=args.computation_tag,
                                       yRange=(-13,1),
                                       comment="",
                                       plotname="comparison_operon_position",
                                    destinations=(h5destination_operon_last, h5destination_operon_1st_and_not_last, h5destination_operon_2nd_and_not_last, h5destination_operon_3rd_and_not_last ))
            
            # numGenesWithShortIntergenicRegions = np.min(h5destination_integenic_short.nativeMeanProfile.counts())
            # numGenesWithLongIntergenicRegions  = np.min(h5destination_integenic_long.nativeMeanProfile.counts())
            
            # plot2WayMFEComparison( taxid,
            #                        profileId,
            #                        pd.DataFrame({'native':   h5destination_integenic_short.getNativeProfile(),
            #                                      'shuffled': h5destination_integenic_short.getShuffledProfile()},
            #                                     index=xRange),
            #                        pd.DataFrame({'native':   h5destination_integenic_long.getNativeProfile(),
            #                                      'shuffled': h5destination_integenic_long.getShuffledProfile()},
            #                                     index=xRange),
            #                        computationTag=args.computation_tag,
            #                        yRange=(-13,1),
            #                        comment="Intergenic region lengths\nShort (N>={}) Long (N>={}) Total (N>={}) ".format( numGenesWithShortIntergenicRegions, numGenesWithLongIntergenicRegions, numGenesWithShortIntergenicRegions + numGenesWithLongIntergenicRegions ) )


            # numGenesLastInOperon     = np.min(h5destination_operon_last.nativeMeanProfile.counts())
            # numGenesNotLastInOperon  = np.min(h5destination_operon_not_last.nativeMeanProfile.counts())
            
            # plot2WayMFEComparison( taxid,
            #                        profileId,
            #                        pd.DataFrame({'native':   h5destination_operon_not_last.getNativeProfile(),
            #                                      'shuffled': h5destination_operon_not_last.getShuffledProfile()},
            #                                     index=xRange),
            #                        pd.DataFrame({'native':   h5destination_operon_last.getNativeProfile(),
            #                                      'shuffled': h5destination_operon_last.getShuffledProfile()},
            #                                     index=xRange),
            #                        computationTag=args.computation_tag,
            #                        yRange=(-13,1),
            #                        comment="Last in operon\nNo (N>={}) Yes (N>={}) Total (N>={}) ".format( numGenesNotLastInOperon, numGenesLastInOperon, numGenesNotLastInOperon + numGenesLastInOperon ) )

            numGenesLastInOperonWithAug     = np.min(h5destination_operon_last_aug.nativeMeanProfile.counts())
            numGenesLastInOperonWithoutAug  = np.min(h5destination_operon_last_no_aug.nativeMeanProfile.counts())

            plotMultipleMFEComparison( taxid,
                                   profileId,
                                       (
                                        (pd.DataFrame({'native':   h5destination_operon_last_aug.getNativeProfile(),
                                                       'shuffled': h5destination_operon_last_aug.getShuffledProfile()},
                                                      index=xRange),
                                         "AUG"),
                                           (pd.DataFrame({'native':   h5destination_operon_last_no_aug.getNativeProfile(),
                                                          'shuffled': h5destination_operon_last_no_aug.getShuffledProfile()},
                                                         index=xRange),
                                            "No AUG")
                                       ),
                                   computationTag=args.computation_tag,
                                   yRange=(-13,1),
                                   comment="Last in operon\nAUG (N>={}) No AUG (N>={}) Total (N>={}) ".format( numGenesLastInOperonWithAug, numGenesLastInOperonWithoutAug, numGenesLastInOperonWithAug+numGenesLastInOperonWithoutAug ),
                                   plotname="comparison_last_aug")
            
            # plot2WayMFEComparison( taxid,
            #                        profileId,
            #                        pd.DataFrame({'native':   h5destination_operon_last_aug.getNativeProfile(),
            #                                      'shuffled': h5destination_operon_last_aug.getShuffledProfile()},
            #                                     index=xRange),
            #                        pd.DataFrame({'native':   h5destination_operon_last_no_aug.getNativeProfile(),
            #                                      'shuffled': h5destination_operon_last_no_aug.getShuffledProfile()},
            #                                     index=xRange),
            #                        computationTag=args.computation_tag,
            #                        yRange=(-13,1),
            #                        comment="Last in operon\nAUG (N>={}) No AUG (N>={}) Total (N>={}) ".format( numGenesLastInOperonWithAug, numGenesLastInOperonWithoutAug, numGenesLastInOperonWithAug+numGenesLastInOperonWithoutAug ) )

            numGenesNotLastInOperonWithAug     = np.min(h5destination_operon_not_last_aug.nativeMeanProfile.counts())
            numGenesNotLastInOperonWithoutAug  = np.min(h5destination_operon_not_last_no_aug.nativeMeanProfile.counts())
            
            plotMultipleMFEComparison( taxid,
                                   profileId,
                                       (
                                        (pd.DataFrame({'native':   h5destination_operon_not_last_aug.getNativeProfile(),
                                                       'shuffled': h5destination_operon_not_last_aug.getShuffledProfile()},
                                                      index=xRange),
                                         "AUG"),
                                           (pd.DataFrame({'native':   h5destination_operon_not_last_no_aug.getNativeProfile(),
                                                          'shuffled': h5destination_operon_not_last_no_aug.getShuffledProfile()},
                                                         index=xRange),
                                            "No AUG")
                                       ),
                                   computationTag=args.computation_tag,
                                   yRange=(-13,1),
                                   comment="Not last in operon\nAUG (N>={}) No AUG (N>={}) Total (N>={}) ".format( numGenesNotLastInOperonWithAug, numGenesNotLastInOperonWithoutAug, numGenesNotLastInOperonWithAug+numGenesNotLastInOperonWithoutAug ),
                                   plotname="comparison_not_last_aug")
            
            
            # numGenesFirstInOperon     = np.min(h5destination_operon_first.nativeMeanProfile.counts())
            # numGenesNotFirstInOperon  = np.min(h5destination_operon_not_first.nativeMeanProfile.counts())
            
            # plot2WayMFEComparison( taxid,
            #                        profileId,
            #                        pd.DataFrame({'native':   h5destination_operon_not_first.getNativeProfile(),
            #                                      'shuffled': h5destination_operon_not_first.getShuffledProfile()},
            #                                     index=xRange),
            #                        pd.DataFrame({'native':   h5destination_operon_first.getNativeProfile(),
            #                                      'shuffled': h5destination_operon_first.getShuffledProfile()},
            #                                     index=xRange),
            #                        computationTag=args.computation_tag,
            #                        yRange=(-13,1),
            #                        comment="First in operon\nNo (N>={}) Yes (N>={}) Total (N>={}) ".format( numGenesNotFirstInOperon, numGenesFirstInOperon, numGenesNotFirstInOperon + numGenesFirstInOperon ) )

            
            
            # numGenesWithShortIntergenicRegions = np.min(h5destination_integenic_short.nativeMeanProfile.counts())
            # numGenesWithOverlaps               = np.min(h5destination_integenic_overlap.nativeMeanProfile.counts())
            
            # plot2WayMFEComparison( taxid,
            #                        profileId,
            #                        pd.DataFrame({'native':   h5destination_integenic_short.getNativeProfile(),
            #                                      'shuffled': h5destination_integenic_short.getShuffledProfile()},
            #                                     index=xRange),
            #                        pd.DataFrame({'native':   h5destination_integenic_overlap.getNativeProfile(),
            #                                      'shuffled': h5destination_integenic_overlap.getShuffledProfile()},
            #                                     index=xRange),
            #                        computationTag=args.computation_tag,
            #                        yRange=(-13,1),
            #                        comment="Intergenic region lengths\nShort (N>={}) Overlapping (N>={}) Total (N>={}) ".format( numGenesWithShortIntergenicRegions, numGenesWithOverlaps, numGenesWithShortIntergenicRegions + numGenesWithOverlaps ) )



            scatterPlot( taxid, profileId, dLFEvsFlankingLength, xvar='flanking_length', yvar='dLFE', colorvar='nextCDSOppositeStrand', title="flanking_length - %s")

            #plotMFEProfileV3(taxid, profileId, df, dLFEData=meanDeltaLFE, ProfilesCount=minCount)
            #plotMFEProfileV3(taxid, profileId, df, dLFEData=meanDeltaLFE, wilcoxon=wilcoxonDf, transitionPeak=guPeakDf, transitionPeakPos=peakPos*10, edgeWilcoxon=edgeWilcoxonDf, ProfilesCount=minCount)

            # Plot the number of genes included in each profile position
            plotXY(
                taxid,
                profileId,
                pd.DataFrame( { "num_genes": h5destination.nativeMeanProfile.counts() }, index=xRange ),
                "position",
                "num_genes",
                "Number of genes included, per starting position (all)",
                computationTag=args.computation_tag
                )
            plotXY(
                taxid,
                profileId,
                pd.DataFrame( { "num_genes_opposite": h5destination_next_cds_opposite.nativeMeanProfile.counts() }, index=xRange ),
                "position",
                "num_genes_opposite",
                "Number of genes included, per starting position (opposite)",
                computationTag=args.computation_tag
                )
            
            # scatterPlotWithKernel(
            #     taxid,
            #     profileId,
            #     geneLevelScatter,
            #     "gc",
            #     "logpval",
            #     "GC vs. MFE selection - %s"
            #     )

            # if( args.codonw ):
            #     scatterPlot(
            #         taxid,
            #         profileId,
            #         geneLevelScatter,
            #         "GC3s",
            #         "logpval",
            #         "GC3s vs. MFE selection - %s"
            #     )

            # if( args.codonw ):
            #     scatterPlot(
            #         taxid,
            #         profileId,
            #         geneLevelScatter,
            #         "gc",
            #         "Nc",
            #         "GC vs. ENc - %s"
            #     )

            # if( args.codonw ):
            #     scatterPlot(
            #         taxid,
            #         profileId,
            #         geneLevelScatter,
            #         "GC3s",
            #         "Nc",
            #         "GC3s vs. ENc - %s"
            #     )

            # if( args.codonw ):
            #     scatterPlot(
            #         taxid,
            #         profileId,
            #         geneLevelScatter,
            #         "Nc",
            #         "logpval",
            #         "ENc vs. MFE selection - %s"
            #     )

            # if( args.codonw ):
            #     scatterPlot(
            #         taxid,
            #         profileId,
            #         geneLevelScatter,
            #         "CBI",
            #         "logpval",
            #         "CBI vs. MFE selection - %s"
            #     )


            # if( args.pax_db ):
            #     #print(geneLevelScatter.head())
            #     scatterPlotWithColor(
            #         taxid,
            #         profileId,
            #         shuffleType,
            #         geneLevelScatter,
            #         "gc",
            #         "logpval",
            #         "pa",
            #         "GC vs. PA - %s"
            #     )

            #     if( args.codonw ):
            #         scatterPlot(
            #             taxid,
            #             profileId,
            #             geneLevelScatter,
            #             "Nc",
            #             "pa",
            #             "ENc vs. PA - %s"
            #         )


                # dfProfileByPA = pd.DataFrame( { "native": nativeMeanProfile.value(), "shuffled": shuffledMeanProfile.value(), "position": xRange, "shuffled25":shuffled25Profile.value(), "shuffled75":shuffled75Profile.value(), "native_pa_high":nativeMeanProfile_HighPAOnly.value(), "native_pa_med":nativeMeanProfile_MediumPAOnly.value(), "native_pa_low":nativeMeanProfile_LowPAOnly.value() }, index=xRange )

                # plotMFEProfileByPA(taxid, profileId, dfProfileByPA)

            # # Try to fit a linear model to describe the gene-level data
            # if( args.codonw ):
            #     if( args.pax_db ):
            #         model = ols("logpval ~ gc + cds_length_nt + Nc + GC3s + CAI + pa", data=geneLevelScatter).fit()
            #     else:
            #         model = ols("logpval ~ gc + cds_length_nt + Nc + GC3s + CAI", data=geneLevelScatter).fit()
            # else:
            #     model = ols("logpval ~ gc + cds_length_nt", data=geneLevelScatter).fit()

            # print(model.params)
            # print(model.summary())
            # print("r     = %f" % model.rsquared**.5)
            # print("r_adj = %f" % model.rsquared_adj**.5)



            spearman_rho = geneLevelScatter.corr(method='spearman')
            print(spearman_rho)
            spearman_rho.to_csv('mfe_v2_spearman_%d_%s_t%d.csv' % (taxid, profileId, shuffleType))



            # vars = ['gc', 'logpval', 'pa', 'cds_length_nt']
            # spearman_rho  = np.zeros((len(vars),len(vars)), dtype=float)
            # spearman_pval = np.zeros((len(vars),len(vars)), dtype=float)
            # for n1, var1 in enumerate(vars):
            #     for n2, var2 in enumerate(vars):
            #         rho, pval = spearmanr(geneLevelScatter[var1], geneLevelScatter[var2], nan_policy='omit')
            #         spearman_rho[n1,n2] = rho
            #         spearman_pval[n1,n2] = pval
            # print(spearman_rho)
            # print(spearman_pval)



            print(statisticsDF)


            # ------------------------------------------------------------------------------------
            # Print final report

            print("Got %d results" % n)

            print(x1)
            print(x2)
            print(x3)

        print("//"*20)
        print(list(combinedData.keys()))

        print("Found stop codons: {}".format(stopCodonFreq))

        if len(combinedData)>1:
            profileId = "%d_%d_%s" % (args.profile[0], args.profile[1], args.profile[2])
            plotMFEProfileForMultipleRandomizations(taxid, profileId, combinedData)

        return (taxid, (x1,x2,x3))


def calcProfilesForSpeciesX(taxid, args):

    # TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY #
    #return (taxid,)
    # TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY ### TEST ONLY #
    
    p = ProfilePlot(taxid, args)
    return p.performPlots()

def runDistributed(args):
    import _distributed
    import dask

    scheduler = _distributed.open()

    results = {}

    taxids = []
    delayedCalls = []

    for taxid in args.taxid:
        call = dask.delayed( calcProfilesForSpeciesX )(taxid, args)
        delayedCalls.append( call )
        taxids.append(taxid)

    futures = scheduler.compute(delayedCalls) # submit all delayed calculations; obtain futures immediately

    try:
        _distributed.progress(futures) # wait for all calculations to complete
    except Exception as e:
        print(E)
    print("\n")

    print("Waiting for all tasks to complete...")
    _distributed.wait(futures)

    results = {}
    errorsCount = 0
    for taxid, f in zip(taxids, futures):
        try:
            r = scheduler.gather(f)
            returnedTaxId = r[0]
            assert(taxid==returnedTaxId)
            results[taxid] = r
            
        except Exception as e:
            print(e)
            results[taxid] = None
            errorsCount += 1

    print("Finished with %d errors" % errorsCount)
    return results
        

if __name__=="__main__":
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--taxid", type=parseList(int))
    argsParser.add_argument("--all-taxa", type=bool, default=False)
    argsParser.add_argument("--profile", type=parseProfileSpec())
    argsParser.add_argument("--computation-tag", type=int, default=Sources.RNAfoldEnergy_SlidingWindow40_v2)
    argsParser.add_argument("--shuffle-types", type=parseList(int) )
    argsParser.add_argument("--num-shuffles", type=int, default=20)
    #argsParser.add_argument("--pax-db", type=str, required=False)
    argsParser.add_argument("--codonw", action="store_true", default=False)
    argsParser.add_argument("--distributed", action="store_true", default=False)
    argsParser.add_argument("--plot-adaptive-lengths", action="store_true", default=False)
    argsParser.add_argument("--utr3prime-length-effect-plot", action="store_true", default=False)
    argsParser.add_argument("--pa", action="store_true", default=False)
    argsParser.add_argument("--dump", action="store_true", default=False)
    args = argsParser.parse_args()

    if( args.all_taxa ):
        args.taxid = list(allSpeciesSource())

    if( args.taxid is None ):
        raise Exception("No species requested (use '--taxid tax1,tax2,tax3' or '--all-taxa')")
    
    # ------------------------------------------------------------------------------------
    # Argument validity checks
    if( len(args.taxid) > len(frozenset(args.taxid)) ):
        raise Exception("Duplicate taxid encountered in list %s" % args.taxid)  # Make sure no taxid was specified twice (will skew calculations...)

    checkSpeciesExist(args.taxid)  # Check for non-existant taxids to avoid doomed runs

    if( args.profile[2] != "begin" and args.profile[2] != "end" and args.profile[2] != "stop3utr" ):
        raise Exception("Unsupported profile reference '%s'" % args.profile[2]) # Currently only profile with reference to CDS 'begin' are implemented...

    results = None
    if( not args.distributed ):
        # run locally
        results = {}
        for taxid in args.taxid:
            ret = calcProfilesForSpeciesX(taxid, args)
            results[taxid] = ret
    else:
        results = runDistributed(args)

    print(results)
    print("Total results: %d" % len(results))
    print("Succeeded: %d" % len([1 for x in list(results.values()) if not x is None]))
    failed = [k for k,v in list(results.items()) if v is None]
    print("Failed: %d (%s)" % (len(failed), failed))
    

            
    
