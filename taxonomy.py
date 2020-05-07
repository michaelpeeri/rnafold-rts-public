from __future__ import division
# Note: use 'export OPENBLAS_NUM_THREADS=1'
#       to limit automatic parallelization done by BLAS (during clustering)
from builtins import map
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
from data_helpers import allSpeciesSource, getSpeciesTemperatureInfo, getSpeciesProperty, CDSHelper, getSpeciesFileName
import re
from csv import DictWriter
from random import randint # Testing only
from pyvirtualdisplay.display import Display  # use Xvnc to provide a headless X session (required by ete for plotting)
from ete3 import Tree, TreeStyle, NodeStyle, TextFace, StaticItemFace, faces, AttrFace, PhyloTree
#from PyQt4 import QtCore, QtGui
import numpy as np
import pandas as pd
from mfe_plots import getHeatmaplotProfilesValuesRange, getProfileHeatmapTile, getLegendHeatmapTile, loadProfileData, loadPhylosignalProfiles, PCAForProfiles, getHeatmaplotProfilesValuesRange, getNodeDiversityPlot, plotProfilesDists, plotPanelizedDLFEComparison
#from reference_trees import pruneTreeByTaxonomy #, extendTreeWithSpecies, getTaxidsFromTree
from ncbi_taxa import ncbiTaxa
from paxdb import getSpeciesPaxdbData
#from cluster_profiles import analyzeProfileClusters, calcDiversityMetrics, correlationMetric, plotDistancesDistribution
#from endosymbionts import isEndosymbiont


# ---------------------------------------------------------
# configuration
# ---------------------------------------------------------
# Length at which to truncate taxon names

maxDisplayNameLength = 27

# Exclusion and inclusion list
speciesToExclude = frozenset(( 405948,946362,470, 158189, 456481, 505682, 1280, 4932, 508771, 2850, 753081, 27923, 6239, 559292, 486041, 237631, 4781, 1257118 )) # ,   272633, 722438 ))  # Hide species with missing data or those that weren't processed successfully 
speciesToInclude = frozenset()  # If non empty, only these species will be included

# Master font scale (points)
fontScale = 16

# Dimensions for drawing profiles
profileDrawingWidth  = 150
profileDrawingHeight = fontScale*1.5
#diversityIconDrawingWidth = 150
diverScale = 85

#
profileStepNt  = 10
profileWidthNt = 40

#
# Collapsed-tree plotting
#
# Distance threshold for deciding how many clusters to use for a given taxon
#max_distance_to_split_clusters = 1.30  # previous level (with 4 levels)
#max_distance_to_split_clusters = 1.60
#max_distance_to_split_clusters = 1.03  # ===> Aquificae  d=1.02908593205
max_distance_to_split_clusters = 0.9
#max_distance_to_split_clusters = 0.9
# Maximum number of clusters (for each taxon)
max_clusters = 18
# Taxons below this level will be merged
collapedTaxonomicTreeLevel = 3
# Font-size for the taxon name at each level
levelFontSizes = {0: 1.5, 1: 2.1, 2:1.8, 3:1.5}
# Display distance between centroids (to help choose max_distance_to_split_clusters)
showInterCentroidDistances = False
# Use fixed Y scale (TODO - use adaptive scale, as in the other profile plots)
yScale = None
# For K-means, choose the number of initial points to use
#n_init=10000
n_init=750


# Phylosignal profiles
#phylosignalProfilesFile = "phylosignalLipaOutputFile.csv"

# ---------------------------------------------------------
# Choose plot format:
#
# single-page (long)
kingdomPageAssignment_SinglePage = {2: 0,     # Bacteria
                                    2759: 0,  # Eukaryotes
                                    2157: 0}  # Archaea
# 2-page
kingdomPageAssignment_2page      = {2: 1,     # Bacteria
                                    2759: 0,  # Eukaryotes
                                    2157: 0}  # Archaea

kingdomPageAssignment = kingdomPageAssignment_2page

# ---------------------------------------------------------



def getSpeciesToInclude():
    count = 0;
    taxa = []
    for taxId in allSpeciesSource():

        # Use black list
        if taxId in speciesToExclude:
            continue # Exclude this taxon

        # Use white list (if defined)
        if speciesToInclude:
            if taxId not in speciesToInclude:
                continue # Exclude this taxon

        # Taxon will be included

        taxa.append(taxId)
        count += 1

    print("Using %d species" % count)
    return taxa
    

# Color names: SVG colors. See: http://etetoolkit.org/docs/latest/reference/reference_treeview.html?highlight=imgface#ete3.SVG_COLORS
#temperatureRangeToColor = {'Mesophilic': 'LightGreen', 'Hyperthermophilic': 'Crimson', 'Thermophilic': 'Orange', 'Psychrophilic': 'Cyan', 'Unknown': 'White'}

#salinityToColor = {'NonHalophilic': 'LemonChiffon', 'Mesophilic': 'LemonChiffon', 'ModerateHalophilic': 'Turquoise', 'ExtremeHalophilic': 'Teal'}

#oxygenReqToColor = {'Aerobic': 'CornflowerBlue', 'Anaerobic': 'LightSalmon', 'Facultative': 'Violet', 'Microaerophilic': 'Aquamarine'}

#Counter({'HostAssociated': 43, 'Aquatic': 32, 'Specialized': 25, 'Multiple': 25, 'Terrestrial': 9})
#habitatToColor = {'HostAssociated': 'LightPink' , 'Aquatic': 'LightSeaGreen', 'Specialized': 'Gold', 'Multiple': 'Silver', 'Terrestrial': 'Sienna'}

#algaeToColor = {'Yes': 'MediumTurquoise', 'No': 'LemonChiffon'}
#endosymbiontToColor = {0: 'White', 1: 'Black'}
modelMatchToColor = {0: '#c9c9c9', 1: '#5870e9'}


def nodeLayoutWithTaxonomicNames(node, tileFunc=None, hideEnvironmentalVars=False, highlightSpecies=frozenset(), numProfileGroups=1, profileData=None):
    nstyle = NodeStyle()
    node.set_style(nstyle)
    
    nstyle["size"] = 0     # hide node symbols

    level = len(node.get_ancestors())
    if not node.is_leaf():
        if( level==1 or level==2 or level==3):
            if node.name:
                name = AttrFace("name", fsize=fontScale)
                ########faces.add_face_to_node(n, node, 0, position="float")
                faces.add_face_to_node(name, node, column=0)

            if ( level==1 ):  # Also add species count
                tf = TextFace(len(node.get_leaves()), fsize=fontScale*1.5)
                faces.add_face_to_node(tf, node, column=1, position="float")

        elif( level==0 ):
            tf = TextFace(len(node.get_leaves()), fsize=fontScale*1.5)
            tf.margin_right = 5
            faces.add_face_to_node(tf, node, column=1, position="float")

    else: #node.is_leaf():
        assert(node.name)

        textColor = "black"
        if node.taxId in highlightSpecies:  # highlight species specified with --highlight-species
            textColor = "IndianRed"
            
        name = AttrFace("name", fsize=fontScale, fstyle="italic", fgcolor=textColor)
        faces.add_face_to_node(name, node, column=0)

        for group in range(numProfileGroups):
            #-------------------------------
            # LFE profile
            #
            tile = None
            if not tileFunc is None:
                print("Getting tile for taxid {} group {}...".format(node.taxId, group))
                tile = tileFunc(node.taxId, group)
                print("...{}".format(tile))

            if not tile is None:
                profileFace = faces.ImgFace(tile, width=profileDrawingWidth, height=profileDrawingHeight, is_url=False) # no support for margin_right?
                profileFace.margin_right = 10
                #profileFace.margin_left  = 10
                faces.add_face_to_node(profileFace, node, column=group+1, aligned=True)
            else:
                print("No tile for node {}".format(node.taxId))

        #proteinCount = getSpeciesProperty(taxId, 'protein-count')
        #if not proteinCount[0] is None:
        #    proteinCount = int(proteinCount)
                
        #genomeSizeMB = getSpeciesProperty(taxId, 'genome-size-mb')
        #if not genomeSizeMB[0] is None:
        #    genomeSizeMB = float(genomeSizeMB)
                    
        #genomicGC = getSpeciesProperty(taxId, 'gc-content')
        #if not genomicGC[0] is None:
        #    genomeicGC = float(genomicGC)

        #-------------------------------
        # genomicGC
        #
        # genomicGC = getSpeciesProperty(node.taxId, 'gc-content')[0]
        # if not genomicGC is None:
        #     genomicGC = float(genomicGC)
        #     genomicGCFace = faces.RectFace( width= old_div((genomicGC-18.0),(73.0-18.0)*50) , height=5, fgcolor="SteelBlue", bgcolor="SteelBlue", label={"text":"%.2g"%genomicGC, "fontsize":8, "color":"Black"} )
        #     genomicGCFace.margin_right = 5
        #     faces.add_face_to_node(genomicGCFace, node, column=2+numProfileGroups-1, aligned=True)

        #-------------------------------
        # Model
        if True:
            modelMatch = profileData.getRTSModelTest(node.taxId)
            if not modelMatch is None:
                modelMatchColor = modelMatchToColor[modelMatch]
                modelMatchFace = faces.RectFace( width=25, height=20, fgcolor=modelMatchColor, bgcolor=modelMatchColor )
                modelMatchFace.margin_right = 5
                faces.add_face_to_node(modelMatchFace, node, column=2+numProfileGroups-1, aligned=True)
        
        #-------------------------------
        # ENC'
        #
        # ENc_prime = getSpeciesProperty(node.taxId, 'ENc-prime')[0]
        # if not ENc_prime is None:
        #     ENc_prime = float(ENc_prime)
        #     ENcPrimeFace = faces.RectFace( width= old_div((ENc_prime-20.0),(64.0-20.0)*50) , height=5, fgcolor="Grey", bgcolor="Grey", label={"text":"%.2g"%ENc_prime, "fontsize":8, "color":"Black"} )
        #     ENcPrimeFace.margin_right = 5
        #     faces.add_face_to_node(ENcPrimeFace, node, column=3+numProfileGroups-1, aligned=True)

        #-------------------------------
        # ENC
        #
        # ENc = getSpeciesProperty(node.taxId, 'ENc')[0]
        # if not ENc is None:
        #     ENc = float(ENc)
        #     ENcFace = faces.RectFace( width= old_div((ENc-20.0),(64.0-20.0)*50) , height=5, fgcolor="Grey", bgcolor="Grey", label={"text":"%.2g"%ENc, "fontsize":8, "color":"Black"} )
        #     ENcFace.margin_right = 5
        #     faces.add_face_to_node(ENcFace, node, column=4+numProfileGroups-1, aligned=True)
            

        # if( not hideEnvironmentalVars):
        #     #-------------------------------
        #     # Temperature
        #     #
        #     (numericalProp, categoricalProp) = getSpeciesTemperatureInfo(node.taxId)
        #     if categoricalProp[0] != "Unknown":
        #         temperatureColor = temperatureRangeToColor[categoricalProp[0]]
        #         temperatureFace = faces.RectFace( width=25, height=10, fgcolor=temperatureColor, bgcolor=temperatureColor )
        #         temperatureFace.margin_right = 5
        #         faces.add_face_to_node(temperatureFace, node, column=5+numProfileGroups-1, aligned=True)

        #     #-------------------------------
        #     # Salinity
        #     #
        #     salinity = getSpeciesProperty(node.taxId, 'salinity')[0]
        #     if salinity is None:
        #         salinity = "Unknown"

        #     if salinity != "Unknown":
        #         salinityColor = salinityToColor[salinity]
        #         salinityFace = faces.RectFace( width=25, height=10, fgcolor=salinityColor, bgcolor=salinityColor )
        #         salinityFace.margin_right = 5
        #         faces.add_face_to_node(salinityFace, node, column=6+numProfileGroups-1, aligned=True)


        #     #-------------------------------
        #     # Oxygen-req
        #     #
        #     oxygenReq = getSpeciesProperty(node.taxId, 'oxygen-req')[0]
        #     if oxygenReq is None:
        #         oxygenReq = "Unknown"

        #     if oxygenReq != "Unknown":
        #         oxygenReqColor = oxygenReqToColor[oxygenReq]
        #         oxygenReqFace = faces.RectFace( width=25, height=10, fgcolor=oxygenReqColor, bgcolor=oxygenReqColor )
        #         oxygenReqFace.margin_right = 5
        #         faces.add_face_to_node(oxygenReqFace, node, column=7+numProfileGroups-1, aligned=True)

        #     #-------------------------------
        #     # Habitat
        #     #
        #     habitat = getSpeciesProperty(node.taxId, 'habitat')[0]
        #     if habitat is None:
        #         habitat = "Unknown"

        #     if habitat != "Unknown":
        #         habitatColor = habitatToColor[habitat]
        #         habitatFace = faces.RectFace( width=25, height=10, fgcolor=habitatColor, bgcolor=habitatColor )
        #         habitatFace.margin_right = 5
        #         faces.add_face_to_node(habitatFace, node, column=8+numProfileGroups-1, aligned=True)

        #     #-------------------------------
        #     # Algae
        #     #
        #     algae = getSpeciesProperty(node.taxId, 'algae')[0]
        #     if algae is None:
        #         algae = "Unknown"

        #     if algae != "Unknown":
        #         algaeColor = algaeToColor[algae]
        #         algaeFace = faces.RectFace( width=25, height=10, fgcolor=algaeColor, bgcolor=algaeColor )
        #         algaeFace.margin_right = 5
        #         faces.add_face_to_node(algaeFace, node, column=9+numProfileGroups-1, aligned=True)

        #     #-------------------------------
        #     # Endosymbionts
        #     #
        #     # endsymbiont = isEndosymbiont( node.taxId )
        #     # print("|{}|".format(endsymbiont))
        #     # endsymbiontColor = endosymbiontToColor[endsymbiont]
        #     # endsymbiontFace = faces.RectFace( width=25, height=10, fgcolor=endsymbiontColor, bgcolor=endsymbiontColor )
        #     # endsymbiontFace.margin_right = 5
        #     # faces.add_face_to_node(endsymbiontFace, node, column=10+numProfileGroups-1, aligned=True)
            
        #     #-------------------------------
        #     # paired fraction
        #     #
        #     pairedFraction = getSpeciesProperty(node.taxId, 'paired-mRNA-fraction')[0]
        #     if not pairedFraction is None:
        #         pairedFraction = float(pairedFraction)
        #         pairedFractionFace = faces.RectFace( width= pairedFraction*50 , height=5, fgcolor="SteelBlue", bgcolor="SteelBlue", label={"text":"%.2g"%(pairedFraction*100), "fontsize":8, "color":"Black"} )
        #         pairedFractionFace.margin_right = 5
        #         faces.add_face_to_node(pairedFractionFace, node, column=11+numProfileGroups-1, aligned=True)

        # Hide the lines of dummyTopology nodes (i.e., nodes for which the tree topology is unknown and were added under the top node)
        if "dummyTopology" in node.features and node.dummyTopology:
            nstyle["hz_line_width"] = 0
            nstyle["hz_line_color"] = "#ffffff"
            nstyle["vt_line_width"] = 0
            nstyle["vt_line_color"] = "#ffffff"

"""
Return unary layout function that also receives a tile function (for getting image to plot along each node)
TODO - are there other parameters to layout funcs that should be passed?
"""
def makeNodeLayoutFuncWithTiles( layoutfn, tileFunc=None, hideEnvironmentalVars=False, highlightSpecies=frozenset(), numProfileGroups=1, profileData=None):
    print("makeNodeLayoutFuncWithTiles: {}".format(len(highlightSpecies)))
    return lambda node: layoutfn(node, tileFunc=tileFunc, hideEnvironmentalVars=hideEnvironmentalVars, highlightSpecies=highlightSpecies, numProfileGroups=numProfileGroups, profileData=profileData)

        




def drawTree(tree, basename, *args, **kw):

    tree.render('%s.pdf' % basename, *args, **kw)
    tree.render('%s.svg' % basename, *args, **kw)
        

def simpleNodeLayoutFunc(node):
    if node.is_leaf():
        if node.name != "n/a":
            l = int(node.name)
            binomicName = ncbiTaxa.get_taxid_translator([l])[l]  # There has to be an easier way to look up names...
            #name = AttrFace("name", fsize=12, fstyle="italic")
            name = TextFace(binomicName, fsize=fontScale*1.5, fstyle="italic")
            faces.add_face_to_node(name, node, column=0)

    else:

        try:
            k = node.testK
            if not k is None:
                #binomicName = ncbiTaxa.get_taxid_translator([k])[k]  # There has to be an easier way to look up names...

                name = TextFace(k, fsize=fontScale*1.5)
                #print("--- %s" % l)
                faces.add_face_to_node(name, node, column=0)

        except AttributeError as e:
            pass
        
        try:
            l = node.testL
            if not l is None:
                binomicName = ncbiTaxa.get_taxid_translator([l])[l]  # There has to be an easier way to look up names...

                name = TextFace(binomicName, fsize=fontScale*1.5)
                #print("--- %s" % l)
                faces.add_face_to_node(name, node, column=1)

        except AttributeError as e:
            pass
        

def collectProfilesFromTree(tree, profiles):
    ret = {}
    for taxId in getTaxidsFromTree(tree):
        if taxId in profiles:
            ret[taxId] = profiles[taxId]

    return ret


class DummyResourceManager(object):
    def __init__(self, *dummy1, **dummy2):  pass
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):  pass
    

def drawTrees(completeTree, prunedTree, args=None):
    import os
    from glob import glob

    print("drawTrees()")

    files = []
    if not args is None and args.use_profile_data:
        files = [[x for x in glob(profilesGlob) if os.path.exists(x)] for profilesGlob in args.use_profile_data]
    
    phylosignalProfiles = None
    if( args.use_phylosignal_data ):
        phylosignalProfiles = loadPhylosignalProfiles( args.use_phylosignal_data )
        
    #print("Loading profile data for %d files..." % len(files))
    #(xdata, ydata, ydata_nativeonly, ydata_shuffledonly, labels, groups, filesUsed, biasProfiles, dfProfileCorrs, summaryStatistics) = loadProfileData(files)
    print("(initializing tile-gen)")
    #tileGenerator = ProfileDataCollection(files, phylosignalProfiles, externalYrange = args.use_Y_range, plotWilcoxon=False, plotWilcoxon50_100=args.plot_wilcoxon)
    tileGenerator = ProfileDataCollection(files, phylosignalProfiles, externalYrange = args.use_Y_range, plotWilcoxon=True, plotWilcoxon50_100=False)

    ts = TreeStyle()
    #ts.mode = "c"
    #ts.arc_start = -180
    #ts.arc_span  =  360
    ts.show_branch_length = False
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.layout_fn = simpleNodeLayoutFunc
    ts.scale = 1500
    #ts.branch_vertical_margin = 10  # Y-axis spacing
    prunedTree.dist = 0.1  # Cut the root distance (since we are displaying a partial tree)

    # PCA
    #    if not args.limit_taxonomy is None:
    #        filtered = findDescendentsOfAncestor(tileGenerator.getTaxIds(), args.limit_taxonomy)
    #        biasProfiles = tileGenerator.getBiasProfiles(profilesGroup=0, taxIdsToInclude = filtered)
    #    else:
    #        biasProfiles = tileGenerator.getBiasProfiles(profilesGroup=0)
    
    profilesForPCA = collectProfilesFromTree(prunedTree, tileGenerator.getBiasProfiles(profilesGroup=0) )

    PCAForProfiles( profilesForPCA, tileGenerator.getYRange(), profilesYOffsetWorkaround=args.profiles_Y_offset_workaround, profileScale=args.profile_scale, fontSize=args.font_size, overlapAction="hide", highlightSpecies=args.highlight_species )

    if args.X_server:
        _disp = Display
    else:
        _disp = DummyResourceManager
    
    with _disp(backend='xvnc') as disp:  # Plotting requires an X session
        for treeToPlot, title in ( (completeTree, 'nmicro_s6'), (prunedTree, 'nmicro_s6_pruned') ):
            h = int(round(len(treeToPlot)*3.5, 0))
            
            if not args.hide_environmental_vars:
                h = int(round(len(treeToPlot)*6.0, 0))
            print("(drawTree {}) h={}".format(title, h))
            drawTree( treeToPlot, title,        tree_style=ts,  w=100, h=h, units="mm")
            #profilesForPCA = collectProfilesFromTree(treeToPlot, biasProfiles )
            #drawPCAforTree( profilesForPCA, [0]*len(profilesForPCA) )
        
        plotSpeciesOnTaxonomicTree(
            tileFunc=tileGenerator.getProfileTileFunc(),
            tree=prunedTree,
            hideLegend=args.hide_legend,
            hideEnvironmentalVars=args.hide_environmental_vars,
            treeScale=1000,
            numProfileGroups=tileGenerator.getNumProfileGroups()
        )
        

        # Display is about to close; how to tell tree to disconnect cleanly? (to prevent "Client Killed" message...)
        
    return 0


"""
Plot each profile separataly, and return the filename of a 'tile' that can be included in other graphs.
"""
class ProfileDataCollection(object):
    #def __init__(self, files, biasProfiles, phylosignalProfiles=None):
    def __init__(self, files, phylosignalProfiles=None, externalYrange=None, profileTilesFirstLineFix=(True, True, True), plotWilcoxon=False, plotWilcoxon50_100=False, loadSequenceProfiles=False):
        if not phylosignalProfiles is None:
            raise Exception("'phylosignalProfiles'  is not supported...")

        self.xdata = []
        self.ydata = []
        self.ydata_nativeonly = []
        self.ydata_shuffleonly = []
        self.label = []
        self.groups = []
        self.filesUsed = []
        self.biasProfiles = []
        self.wilcoxonDLFEZeroData = []
        self.wilcoxonDLFE50_100Data = []
        self.dfProfileCorrs = []
        self.summaryStatistics = []
        self.numProfileGroups = len(files)
        self.profileTilesFirstLineFix = profileTilesFirstLineFix
        self.plotWilcoxon = plotWilcoxon
        self.plotWilcoxon50_100 = plotWilcoxon50_100
        self.loadSequenceProfiles = loadSequenceProfiles
        self.sequenceNativeProfiles = []
        self.sequenceRandomizedProfiles = []

        self.bdqaPlottedSpecies = set()

        if externalYrange is None:
            yrange = [0.0, 0.0]
        else:
            yrange = [-externalYrange, externalYrange]
        
        for group, files in enumerate(files):
            (xdata, ydata, ydata_nativeonly, ydata_shuffledonly, labels, groups, filesUsed, biasProfiles, dfProfileCorrs, summaryStatistics, wilcoxonDLFEZeroData, wilcoxonDLFE50_100Data, sequenceNativeProfiles, sequenceRandomizedProfiles) = loadProfileData(files)
            self.xdata.append( xdata )
            self.ydata.append( ydata )
            self.ydata_nativeonly.append( ydata_nativeonly )
            self.ydata_shuffleonly.append( ydata_shuffledonly )
            self.label.append( labels )
            self.groups.append( groups )
            self.filesUsed.append( filesUsed )
            self.biasProfiles.append( biasProfiles )
            self.dfProfileCorrs.append( dfProfileCorrs )
            self.wilcoxonDLFEZeroData.append( wilcoxonDLFEZeroData )
            self.wilcoxonDLFE50_100Data.append( wilcoxonDLFE50_100Data )
            if self.loadSequenceProfiles: 
                self.sequenceNativeProfiles.append( sequenceNativeProfiles )
                self.sequenceRandomizedProfiles.append( sequenceRandomizedProfiles )
                
            self.summaryStatistics.append( summaryStatistics )
            if externalYrange is None:
                yrangeForThisGroup = getHeatmaplotProfilesValuesRange(biasProfiles)  # plot all profiles once, to determine a single color-scale that will fit them all
                if yrangeForThisGroup[0] < yrange[0]:
                    yrange[0] = yrangeForThisGroup[0]
                if yrangeForThisGroup[1] > yrange[1]:
                    yrange[1] = yrangeForThisGroup[1]

        assert(len(self.biasProfiles) == self.numProfileGroups)

        self._yrange = tuple(yrange)

        self._phylosignalProfiles = phylosignalProfiles

        self.allRTSModelResults = {}

    def getProfileTile(self, taxid, profilesGroup=0):
        #print("[--- {} (g{}) ---]".format( self.profileTilesFirstLineFix[profilesGroup], profilesGroup ))

        self.bdqaPlottedSpecies.add( taxid)
        
        if self.plotWilcoxon:
            return getProfileHeatmapTile(taxid, self.biasProfiles[profilesGroup], self._yrange, phylosignalProfiles=self._phylosignalProfiles, profilesGroup=profilesGroup, profileTilesFirstLineFix=self.profileTilesFirstLineFix[profilesGroup], wilcoxonDLFEZero=self.wilcoxonDLFEZeroData[profilesGroup], wilcoxonDLFE50_100=None )
        
        elif self.plotWilcoxon50_100:
            return getProfileHeatmapTile(taxid, self.biasProfiles[profilesGroup], self._yrange, phylosignalProfiles=self._phylosignalProfiles, profilesGroup=profilesGroup, profileTilesFirstLineFix=self.profileTilesFirstLineFix[profilesGroup], wilcoxonDLFEZero=None, wilcoxonDLFE50_100=self.wilcoxonDLFE50_100Data[profilesGroup] )
        
        else:
            return getProfileHeatmapTile(taxid, self.biasProfiles[profilesGroup], self._yrange, phylosignalProfiles=self._phylosignalProfiles, profilesGroup=profilesGroup, profileTilesFirstLineFix=self.profileTilesFirstLineFix[profilesGroup], wilcoxonDLFEZero=None, wilcoxonDLFE50_100=None )

    def getScaleTile(self):
        return getLegendHeatmapTile(  self._yrange )
    """
    Return a free function that will return a tile based on the taxid
    """
    def getProfileTileFunc(self):
        return lambda x, profilesGroup=0: self.getProfileTile(x, profilesGroup)
                
    def getYRange(self):
        return self._yrange

    def getNumProfileGroups(self):
        return self.numProfileGroups

    def getBiasProfiles(self, profilesGroup=0, taxIdsToInclude=None):
        if taxIdsToInclude is None:
            return self.biasProfiles[profilesGroup]

        ret = {}
        include = frozenset(taxIdsToInclude)
        for k,v in list(self.biasProfiles[profilesGroup].items()):
            if k in include:
                ret[k] = v
        return ret

    def getTaxIds(self, profilesGroup=0):
        return list(self.biasProfiles[profilesGroup].keys())

    def getRTSModelTest(self, taxId, profilesGroup=0):
        # if taxId in self.biasProfiles[profilesGroup]:
        #     profile = self.biasProfiles[profilesGroup][taxId]
        #     #profile.loc[-5:20]
        # else:
        #     return None

        if taxId in self.wilcoxonDLFE50_100Data[profilesGroup]:
            wilx = self.wilcoxonDLFE50_100Data[profilesGroup][taxId]
        else:
            return None

        dataInRange = wilx.loc[wilx['pos'].between(-5, 20), :]
        pointMatches = np.logical_and(
            dataInRange.pval <= 1e-2,
            dataInRange.sign == -1 )

        result = pointMatches.astype(int).rolling(3).sum().max() >= 3 # require 3 consecutive matching points

        self.allRTSModelResults[taxId] = result
        
        return result

    def getRTSStatistics(self):
        N         = len(self.allRTSModelResults)
        positives = sum(self.allRTSModelResults.values())
        if N == 0:
            return (0, None, None)
        else:
            return (N, positives, float(positives/N))

    def printStatisticts(self):
        print("==========================================")
        print("Statistics:")
        #self.biasProfiles[profilesGroup]
        for n, group in enumerate( self.biasProfiles ):
            print("Profiles group {}: N={}".format( n, len(group) ))
            print("Species: {}".format(group.keys()))

        print("Species plotted on tree (N={}): {}".format( len(self.bdqaPlottedSpecies), self.bdqaPlottedSpecies ))
        print("Species not plotted: {}".format( frozenset(group.keys()) - self.bdqaPlottedSpecies ))
            
        

def filterTaxidsByParent(taxa, parent):
    ret = []
    
    for taxid in taxa:
        lineage = frozenset(ncbiTaxa.get_lineage(taxid))
    
        if parent in lineage:
            ret.append(taxid)
            
    return ret

        
"""
Plot "statistical" tree, with species names and counts 
This tree should illustrate the included species
"""
def plotSpeciesOnTaxonomicTree(tileFunc=None, tree=None, phylosignalFile="", hideLegend=False, hideEnvironmentalVars=False, ownXserver=False, treeScale=100, highlightSpecies=frozenset(), numProfileGroups=1, limitTaxonomy=None, profileData=None ):
    taxa = getSpeciesToInclude()

    if( not limitTaxonomy is None ):  # limit taxonomy to specified taxon
        taxa = filterTaxidsByParent( taxa, limitTaxonomy  )

    if tree is None:
        # Get the smallest "taxonomic" (i.e., n-ary) tree that includes all specified species
        # See: http://etetoolkit.org/docs/3.0/tutorial/tutorial_ncbitaxonomy.html
        tree = ncbiTaxa.get_topology(taxa, intermediate_nodes=True)

        
    
    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_scale = False
    ts.scale = treeScale
    #ts.branch_vertical_margin = 10  # Y-axis spacing
    ts.layout_fn = makeNodeLayoutFuncWithTiles( nodeLayoutWithTaxonomicNames, tileFunc, hideEnvironmentalVars=hideEnvironmentalVars, highlightSpecies=highlightSpecies, numProfileGroups=numProfileGroups, profileData=profileData  )

    # Add header (with a caption above each column)
    for group in range(numProfileGroups):
        ts.aligned_header.add_face(TextFace("LFE-{}".format(group)), column=group+1)
    ts.aligned_header.add_face(TextFace("GC%"),  column=2+numProfileGroups-1)
    ts.aligned_header.add_face(TextFace("ENc'"), column=3+numProfileGroups-1)
    ts.aligned_header.add_face(TextFace("ENc"),  column=4+numProfileGroups-1)
    if( not hideEnvironmentalVars):
        ts.aligned_header.add_face(TextFace("Tmp"),    column=5+numProfileGroups-1)
        ts.aligned_header.add_face(TextFace("Hal"),    column=6+numProfileGroups-1)
        ts.aligned_header.add_face(TextFace("Oxy"),    column=7+numProfileGroups-1)
        ts.aligned_header.add_face(TextFace("Hab"),    column=8+numProfileGroups-1)
        ts.aligned_header.add_face(TextFace("Alg"),    column=9+numProfileGroups-1)
        ts.aligned_header.add_face(TextFace("Endo"),   column=10+numProfileGroups-1)
        ts.aligned_header.add_face(TextFace("Paired"), column=11+numProfileGroups-1)

    
    if( not hideLegend ):
        # Draw legend
        for cat in ('Psychrophilic', 'Mesophilic', 'Thermophilic', 'Hyperthermophilic'):
            color = temperatureRangeToColor[cat]
            ts.legend.add_face(faces.RectFace( width=100, height=20, fgcolor=color, bgcolor=color, label={"text":cat, "color":"Black", "fontsize":8}), column=1)

        for cat in ('NonHalophilic', 'ModerateHalophilic', 'ExtremeHalophilic'):
            color = salinityToColor[cat]
            ts.legend.add_face(faces.RectFace( width=100, height=20, fgcolor=color, bgcolor=color, label={"text":cat, "color":"Black", "fontsize":8}), column=2)

        for cat in ('Aerobic', 'Facultative', 'Anaerobic', 'Microaerophilic'):
            color = oxygenReqToColor[cat]
            ts.legend.add_face(faces.RectFace( width=100, height=20, fgcolor=color, bgcolor=color, label={"text":cat, "color":"Black", "fontsize":8}), column=3)

        for cat in ('Aquatic', 'Terrestrial', 'HostAssociated', 'Specialized', 'Multiple'):
            color = habitatToColor[cat]
            ts.legend.add_face(faces.RectFace( width=100, height=20, fgcolor=color, bgcolor=color, label={"text":cat, "color":"Black", "fontsize":8}), column=4)

        for cat in ('Yes', 'No'):
            color = algaeToColor[cat]
            ts.legend.add_face(faces.RectFace( width=100, height=20, fgcolor=color, bgcolor=color, label={"text":cat, "color":"Black", "fontsize":8}), column=5)


        
    # Annotate tree nodes
    for node in tree.traverse():
        if not node.name:
            if( node.is_leaf() ):
                print("Warning: node.name missing for node {}".format(node.taxId))
            continue
        
        taxId = int(node.name)
        binomicName = ncbiTaxa.get_taxid_translator([taxId])[taxId]  # There has to be an easier way to look up names...
        node.add_features(displayName=binomicName, taxId=taxId)
        if( len(binomicName) > maxDisplayNameLength):
            node.name = binomicName[:maxDisplayNameLength-1].rstrip() + '...'
        else:
            node.name = binomicName

    
    n = 0
    nsLight = NodeStyle()
    #nsLight["bgcolor"] = "LightSt"
    nsDark  = NodeStyle()
    nsDark["bgcolor"] = "WhiteSmoke"
    for node in tree.traverse():
        if not node.is_leaf():
            continue
        
        if n%2==0:
            node.set_style(nsDark)
        else:
            node.set_style(nsLight)
        n += 1

    if ownXserver:
        _disp = Display
    else:
        _disp = DummyResourceManager

    with _disp(backend='xvnc') as disp:  # Plotting requires an X session

        if not hideEnvironmentalVars:
            h = int(round(len(tree)*2.5, 0))
        else:
            h = int(round(len(tree)*4.0, 0))
        
        tree.render('alltaxa.pdf', tree_style=ts, w=100, h=h, units="mm")
        tree.render('alltaxa.svg', tree_style=ts, w=100, h=h, units="mm")

        print("------------------ Ignore error message ------------------")
        # Display is about to close; how to tell tree to disconnect cleanly? (to prevent "Client Killed" message...)

    return 0


def makeProfilesArray(keys, biasProfiles):
    if( not keys ):
        raise Exception("No keys specified")

    # Filter keys with missing profiles
    keysFound = [key for key in keys if key in biasProfiles]
    if( not keysFound ):
        #return None
        raise Exception("No profiles found matching requested keys: %s..." % str(keys))

    # Find out the length of the profile
    profileLength = biasProfiles[keysFound[0]].shape[0]  # We will use the length of the first profile, and later verify the following profiles match
    assert(profileLength > 3)

    # Copy the profiles into an array
    out = np.zeros( (len(keysFound), profileLength ) )
    for i, key in enumerate(keysFound):
        out[i,:] = biasProfiles[key]

    return out

# def nodeLayoutForCollapsedTree(node, groupMembers, profileDataCollection): #biasProfiles, tileFunc=None):
#     level = len(node.get_ancestors())
#     taxId = int(node.name)
#     nodeName = ncbiTaxa.get_taxid_translator([taxId])[taxId]  # There has to be an easier way to look up names...
#     members = groupMembers[taxId]

#     nstyle = NodeStyle()
#     nstyle["size"] = 0     # hide node symbols
#     node.set_style(nstyle)
        
#     if level > 0:
#         name = TextFace(nodeName, fsize=fontScale*levelFontSizes[level])
#         faces.add_face_to_node(name, node, column=0)

        
#     if( level >= collapedTaxonomicTreeLevel - 2 ):
#         assert(node.name)

#         tile = None

#         profilesArray = makeProfilesArray( members, profileDataCollection.getBiasProfiles( profilesGroup=0 ) )  # TODO add groups
#         if profilesArray is None:
#             pass

#         #print("//// %d %s" % (level, profilesArray.shape))

#         elif( profilesArray.shape[0] == 1 ):
#             #if not taxId in biasProfiles:
#             #    print("TaxId %d not found" % taxId)
                
#             tileDummyId = 1900000000+taxId*10

#             tile = getProfileHeatmapTile(tileDummyId, {tileDummyId:profilesArray[0]}, yScale )

#             if not tile is None:
#                 profileFace = faces.ImgFace(tile, width=profileDrawingWidth, height=profileDrawingHeight, is_url=False)
#                 profileFace.margin_top = 4   # separate this taxon's profiles from the next
#                 faces.add_face_to_node(profileFace, node, column=1 )#, aligned=True)

#                 countFace = faces.TextFace( "1", fsize=fontScale )
#                 countFace.background.color = "#dfdfdf"
#                 countFace.margin_right = 3
#                 faces.add_face_to_node(countFace, node, column=2 )#, aligned=True)

#             else:
#                 print("No node <- Level: %d profilesArray: %s" % (level, profilesArray.shape))
                
#         else:
#             # Perform clustering
#             # centers, labels, dist, distThreshold = analyzeProfileClusters( profilesArray,
#             #                                                                method="KMeans",
#             #                                                                n_init=n_init,
#             #                                                                max_permissible_distance_centroid=max_distance_to_split_clusters,
#             #                                                                max_clusters=max_clusters )  # use default metric
#             centers, labels, dist, distThreshold = analyzeProfileClusters( profilesArray,
#                                                                            method="AggClus",
#                                                                            distThreshold=max_distance_to_split_clusters )
#             print("===> {}  d={}".format( nodeName, dist ) )
#             groupCounts = [sum([1 for x in labels if x==i]) for i in range(len(centers))]  # count how many profiles belong to each group

#             #diversityTile = getNodeDiversityPlot(distThreshold, dist, taxId)

#             #diversityFace = faces.ImgFace( diversityTile, width=diversityIconDrawingWidth, height=diversityIconDrawingWidth, is_url=False )
#             #faces.add_face_to_node( diversityFace, node, 0, position="float" )

#             diversityMetric = calcDiversityMetrics( profilesArray, metric=correlationMetric, saveHistogramAs="diversity_{}_taxid{}.pdf".format(nodeName.replace("/","").replace(":",""), taxId) )
#             #diversityMetric = calcDiversityMetrics( profilesArray )
#             print("Diversity(taxId={}) -> {}".format(taxId, diversityMetric) )
#             df1 = faces.CircleFace( radius=diversityMetric*diverScale, color="RoyalBlue" )
#             faces.add_face_to_node( df1, node, 0, aligned=False, position="float-behind" )
#             df1.opacity=0.3

            
#             for i in range(centers.shape[0]):
#                 tileDummyId = 1900000000+taxId*10+i

#                 tile = getProfileHeatmapTile(tileDummyId, {tileDummyId:centers[i]}, yScale )

#                 if not tile is None:
#                     profileFace = faces.ImgFace(tile, width=profileDrawingWidth, height=profileDrawingHeight, is_url=False)
#                     profileFace.margin_bottom  = 2   # separate this taxon's profiles from the next
#                     if i==0:
#                         profileFace.margin_top = 8   # separate this taxon's profiles from the next
#                     faces.add_face_to_node(profileFace, node, column=1 )#, aligned=True)

#                     countFace = faces.TextFace( groupCounts[i], fsize=fontScale*0.95 )
#                     countFace.background.color = "#dfdfdf"
#                     countFace.margin_right = 3
#                     countFace.margin_top = 0
#                     countFace.margin_bottom = 0
#                     faces.add_face_to_node(countFace, node, column=2 )#, aligned=True)

#                     if showInterCentroidDistances:
#                         distFace = faces.TextFace( "%.4g" % dist, fsize=fontScale )
#                         #distFace.background.color = "#dfdfdf"
#                         distFace.margin_left = 7
#                         distFace.margin_right = 3
#                         faces.add_face_to_node(distFace, node, column=3 )#, aligned=True)

#                 else:
#                     print("No node <- Level: %d  Id: %d profilesArray: %s" % (level, i, profilesArray.shape))

#     else:
#         pass
#         #countFace = faces.TextFace( len(members), fsize=fontScale*1.5 )
#         #countFace.background.color = "#dfdfdf"
#         #countFace.margin_right = 3
#         #faces.add_face_to_node(countFace, node, column=1 )#, aligned=True)
        
        


# def plotCollapsedTaxonomicTree(profileDataCollection, ownXserver=False):
#     allTaxa = getSpeciesToInclude()

#     # ---------------------------------------------------------
#     collapsedTaxa = set()
#     groupMembers = {}
    
#     for taxid in allTaxa:
#         lineage = ncbiTaxa.get_lineage(taxid)

#         group = lineage[collapedTaxonomicTreeLevel]

#         # Create a mapping containing all leaves found under a grouping taxon
#         for ancestor in lineage[:collapedTaxonomicTreeLevel+1]:
#             if ancestor in groupMembers:
#                 groupMembers[ancestor].append( taxid )
#             else:
#                 groupMembers[ancestor] = [taxid]

#         collapsedTaxa.add(group)


#     def filterTaxaByPage(taxa, pageToInclude):
#         out = set()
#         for taxid in taxa:
#             lineage = ncbiTaxa.get_lineage(taxid)
#             taxonPage = kingdomPageAssignment[lineage[2]]
#             if taxonPage == pageToInclude:
#                 out.add(taxid)
#         return out

#     global yScale
#     if yScale is None:
#         yScale = profileDataCollection.getYRange()

#     if ownXserver:
#         _disp = Display
#     else:
#         _disp = DummyResourceManager
    
#     with _disp(backend='xvnc') as disp:  # Plotting requires an X session
        
#         def plotCollapsedTree(taxa, i):
#             print("Page %d (%d taxons)" % (i, len(taxa)))
            
#             # Get the smallest "taxonomic" (i.e., n-ary) tree that includes all specified species
#             # See: http://etetoolkit.org/docs/3.0/tutorial/tutorial_ncbitaxonomy.html
#             # In this case, we provide mid-level taxons to get the "collapsed" tree
#             tree = ncbiTaxa.get_topology(taxa, intermediate_nodes=True)


#             # Configure tree plotting settings
#             ts = TreeStyle()
#             ts.show_leaf_name = False
#             ts.branch_vertical_margin = 6  # Y-axis spacing
#             ts.layout_fn = lambda node: nodeLayoutForCollapsedTree(node, groupMembers, profileDataCollection)
#             ts.show_scale = False
#             ts.margin_top = 25
#             ts.margin_bottom = 50

#             # Get all the profiles from group 0 (i.e., those matching the first glob expression)
#             profiles = profileDataCollection.getBiasProfiles( profilesGroup=0 )

#             # Plot distribution of all pairwise distances (diagnostic)
#             plotDistancesDistribution( makeProfilesArray( list(profiles.keys()), profiles ), savePlotAs="corrDist.pdf" )

#             # Create figure legend
#             #if i==max(kingdomPageAssignment.values()):  # only on last page
#             if i==0:                                     # only on first page

#                 ts.legend_position = 1 #=top left

#                 lastProfileStart = (profiles[list(profiles.keys())[0]].shape[0] - 1) * profileStepNt
#                 widthFor100nt = profileDrawingWidth * (100. / lastProfileStart)
#                 scaleFace = faces.RectFace( width=widthFor100nt, height=profileDrawingHeight, fgcolor="black", bgcolor="black" )
#                 scaleFace.margin_top     = 10
#                 scaleFace.margin_bottom  = 5
#                 scaleFace.margin_left    = 20
#                 scaleFace.margin_right   = 50
#                 ts.legend.add_face( scaleFace, column=0 )
#                 scaleFaceText = TextFace("100 bp", fsize=fontScale*1.5 )
#                 scaleFaceText.margin_top = 0
#                 ts.legend.add_face( scaleFaceText, column=0 )
                
#                 widthForWindow = profileDrawingWidth * (float(profileWidthNt) / lastProfileStart)
#                 windowWidthFace = faces.RectFace( width=widthForWindow, height=profileDrawingHeight, fgcolor="black", bgcolor="black" )
#                 windowWidthFace.margin_top    = 10
#                 windowWidthFace.margin_bottom = 5
#                 windowWidthFace.margin_left   = 20
#                 windowWidthFace.margin_right  = 50
#                 ts.legend.add_face( windowWidthFace, column=1 )
#                 windowWidthFace = TextFace("40 bp", fsize=fontScale*1.5 )
#                 windowWidthFace.margin_top    = 0
#                 ts.legend.add_face( windowWidthFace, column=1 )
                
#                 # Draw color scale
                
#                 #ts.legend.add_face( TextFace(unichr(0x1d6ab)+"LFE", fsize=12    ), column=0 )
#                 #ts.legend.add_face( TextFace("\xf0\x9d\x9a\xab"+"LFE", fsize=12    ), column=0 )
#                 dLFEFace = TextFace("dLFE", fsize=fontScale*2.0    )
#                 dLFEFace.margin_bottom = 0
#                 ts.legend.add_face( dLFEFace, column=2 )

#                 unitsFace = TextFace("kcal/mol/window", fsize=fontScale*1.5 )
#                 unitsFace.margin_top = 0
#                 ts.legend.add_face( unitsFace, column=2 )

#                 legendColorScaleTile = getLegendHeatmapTile(yScale)
#                 legendColorScaleFace = faces.ImgFace(legendColorScaleTile, width=480, height=80, is_url=False) # no support for margin_right?
#                 ts.legend.add_face( legendColorScaleFace, column=3 )

#                 df2 = faces.CircleFace( radius=max_distance_to_split_clusters*diverScale, color="RoyalBlue" )
#                 #df2 = faces.CircleFace( radius=10, color="RoyalBlue" )
#                 ts.legend.add_face( df2, column=4 )
#                 df2.opacity=0.3
                

#                 # Test rich text plotting, using Qt (this didn't work...)
#                 #txtItem = QtGui.QGraphicsTextItem( "Hello, World! " + unichr(0x1d6ab)+ "LFE" )
#                 #txtItem = QtGui.QGraphicsSimpleTextItem()
#                 #txtItem.setPlainText( QtCore.QString( "Hello, World! " ) )   # <-- This step crashes the interpreter silently
                
#                 #siFace = StaticItemFace(txtItem)
#                 #siFace.width = 100
#                 #siFace.height = 20
#                 #ts.legend.add_face( siFace, column=5 )

#             # Render and save the tree
#             if collapedTaxonomicTreeLevel==4:
#                 h = int( round(  (len(tree)             *4.5), 0) )
#             else:
#                 h = int( round( ((len(tree) + (i==0)*3 )*6.5), 0) )
                
#             tree.render('alltaxa.collapsed.%d.pdf' % i, tree_style=ts, w=100, h=h, units="mm")
#             tree.render('alltaxa.collapsed.%d.svg' % i, tree_style=ts, w=100, h=h, units="mm")

#         # Plot each page (with a different subset of the tree)
#         for pageNum in sorted(set(kingdomPageAssignment.values())):
#             plotCollapsedTree(filterTaxaByPage(collapsedTaxa, pageNum), pageNum)
    
#     print("------------------ Ignore error message ------------------")
#     # Display is about to close; how can we disconnect cleanly? (to prevent "Client Killed" message...)

#     return 0

def outputBDQAtables( indexAndNamePairs, taxId ):
    for t in indexAndNamePairs:
        assert(len(t[0])==len(indexAndNamePairs[0][0]))
    
    with open("BDQA_{}_{}.csv".format( getSpeciesFileName(taxId), indexAndNamePairs[0][1] ), "w", newline='') as csvfile:
        writer = DictWriter( csvfile, fieldnames=tuple([x[1] for x in indexAndNamePairs]) )
        writer.writeheader()

        for n in range(len( indexAndNamePairs[0][0])):
            writer.writerow( dict( tuple( [(x[1], x[0][n]) for x in indexAndNamePairs] ) ) )
    



def createTraitMapping(trait):
    ret = {}
    for taxId in allSpeciesSource():

        if trait=="GC":
            raise Exception("NotImpl")
            #genomicGC = getSpeciesProperty(taxId, 'gc-content')[0]
            #if not genomicGC is None:
            #    ret[taxId] = float(genomicGC)
                
        elif trait=="Temp":
            optimumTemp = getSpeciesProperty(taxId, 'optimum-temperature')[0]
            if not optimumTemp is None:
                ret[taxId] = float(optimumTemp)
                
        # elif trait=="Endosymbiont":
        #     endsymbiont = isEndosymbiont( taxId )
        #     if not endsymbiont is None:
        #         ret[taxId] = endsymbiont
                
        else:
            raise Exception("Unknown trait {}".format(trait))
                            
    return ret
        

def plotProfilesDistribution( args, profiles ):
    allProfiles = profiles.getBiasProfiles()

    dfProfiles                      = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesNextCDSSameStrand     = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesNextCDSOppositeStrand = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesNextCDSOverlapping    = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesNextCDSDistUnder25    = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesNextCDSDistOver25     = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesOperonLast            = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesOperon1stAndNotLast   = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesOperon2ndAndNotLast   = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesOperon3rdAndNotLast   = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesPALow                 = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesPAMed                 = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesPAHigh                = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesStopUGA               = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesStopUAA               = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfProfilesStopUAG               = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )
    dfStdProfiles                   = pd.DataFrame( columns=range(-100,100), index=pd.Index([]) )

    for taxId, currProfile in allProfiles.items():
        if not args.limit_taxonomy is None:
            lineage = frozenset(ncbiTaxa.get_lineage(taxId))

            if not args.limit_taxonomy in lineage:
                continue
        
        dfProfiles = dfProfiles.append( pd.DataFrame( dict(zip( currProfile.index.values, currProfile.values )), index=pd.Index([taxId]), columns=currProfile.index.values ) )  # convert series (column) into row

        if taxId in profiles.sequenceNativeProfiles[0]:
            #
            # To enable grouping of genes by different categories, find all genes that belong to each of the categories
            nativeProfiles     = profiles.sequenceNativeProfiles[0][taxId]
            randomizedProfiles = profiles.sequenceRandomizedProfiles[0][taxId]
            assert(np.all( nativeProfiles.index.values == randomizedProfiles.index.values ))

            oppStrandIndex            = []
            sameStrandIndex           = []
            nextCDSOverallapingIndex  = []
            nextCDSDistUnder25Index   = []
            nextCDSDistOver25Index    = []
            operonLastIndex           = []
            operon1stAndNotLastIndex  = []
            operon2ndAndNotLastIndex  = []
            operon3rdAndNotLastIndex  = []
            operonProtIdIndex         = []
            operonGeneIdIndex         = []
            paHighIndex               = []
            paMedIndex                = []
            paLowIndex                = []
            dfProfilesStopUGAIndex    = []
            dfProfilesStopUAAIndex    = []
            dfProfilesStopUAGIndex    = []
            for protId in nativeProfiles.index.values:
                cds = CDSHelper( taxId, protId )
                #
                # Prepare data by next CDS strand
                isNextCDSOnOppositeStrand = cds.nextCDSOnOppositeStrand()
                oppStrandIndex.append(      isNextCDSOnOppositeStrand )
                sameStrandIndex.append( not isNextCDSOnOppositeStrand )

                #
                # Prepare data by intergenic distance
                flanking3UTRRegionLengthNt = cds.flankingRegion3UtrLength()
                nextCDSOverallapingIndex.append( flanking3UTRRegionLengthNt < 0 )
                nextCDSDistUnder25Index.append(  flanking3UTRRegionLengthNt >= 0 and flanking3UTRRegionLengthNt <= 25 )
                nextCDSDistOver25Index.append(   flanking3UTRRegionLengthNt >  25 )

                seq = cds.sequence()
                stopCodonPos = cds.CDSlength()
                stopCodon = seq[stopCodonPos-3:stopCodonPos]
                assert( len(stopCodon) == 3 )
                #
                # Prepare data by stop-codon
                if not stopCodon in ('tga', 'taa', 'tag'):
                    print("Warning: unexpected stop-codon {}".format( stopCodon ))
                    
                dfProfilesStopUGAIndex.append( stopCodon=='tga' )
                dfProfilesStopUAAIndex.append( stopCodon=='taa' )
                dfProfilesStopUAGIndex.append( stopCodon=='tag' )
                
                #
                # Prepare operon data
                oinf = cds.getOperonInfo()
                if (not oinf is None):
                    operonPosFromStart = oinf[0]
                    operonPosFromEnd   = oinf[1]-1 - oinf[0]
                    assert(operonPosFromStart >= 0)
                    assert(operonPosFromEnd >= 0)

                    operonProtIdIndex.append( protId )
                    operonGeneIdIndex.append( cds.getGeneId() )
                    operonLastIndex.append(          operonPosFromEnd == 0 )
                    operon1stAndNotLastIndex.append( operonPosFromEnd != 0 and operonPosFromStart == 0 )
                    operon2ndAndNotLastIndex.append( operonPosFromEnd != 0 and operonPosFromStart == 1 )
                    operon3rdAndNotLastIndex.append( operonPosFromEnd != 0 and operonPosFromStart == 2 )
                else:
                    operonProtIdIndex.append( None )
                    operonGeneIdIndex.append( None )
                    operonLastIndex.append(          False )
                    operon1stAndNotLastIndex.append( False )
                    operon2ndAndNotLastIndex.append( False )
                    operon3rdAndNotLastIndex.append( False )
                    

                #
                # Prepare PA data
                gid = cds.getGeneId()
                proteinAbundanceRel = getSpeciesPaxdbData( taxId, convertToPercentiles=True  )
                PArel = proteinAbundanceRel.get( gid, None )
                
                if not PArel is None:
                    assert( PArel >= 0.0 and PArel <= 1.0 )
                    paHighIndex.append( PArel >= 0.678 )
                    paMedIndex.append(  PArel >= 0.333 and PArel < 0.678 )
                    paLowIndex.append(  PArel <  0.333 )
                else:
                    paHighIndex.append( False )
                    paMedIndex.append(  False )
                    paLowIndex.append(  False )
                    
                    

                
                

            #
            # ...now that the groups are known, calculate a profile for each group (including only the sequences belonging to it)
            oppStrandProfile = np.mean( nativeProfiles.loc[ oppStrandIndex, : ] - randomizedProfiles.loc[ oppStrandIndex, : ], axis=0 )
            dfProfilesNextCDSOppositeStrand = dfProfilesNextCDSOppositeStrand.append( pd.DataFrame( dict(zip( currProfile.index.values, oppStrandProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )
            
            sameStrandProfile = np.mean( nativeProfiles.loc[ sameStrandIndex, : ] - randomizedProfiles.loc[ sameStrandIndex, : ], axis=0 )
            dfProfilesNextCDSSameStrand = dfProfilesNextCDSSameStrand.append( pd.DataFrame( dict(zip( currProfile.index.values, sameStrandProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )

            nextCDSOverlappingProfile = np.mean( nativeProfiles.loc[ nextCDSOverallapingIndex, : ] - randomizedProfiles.loc[ nextCDSOverallapingIndex, : ], axis=0 )
            dfProfilesNextCDSOverlapping = dfProfilesNextCDSOverlapping.append( pd.DataFrame( dict(zip( currProfile.index.values, nextCDSOverlappingProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )

            nextCDSDistanceUnder25Profile = np.mean( nativeProfiles.loc[ nextCDSDistUnder25Index, : ] - randomizedProfiles.loc[ nextCDSDistUnder25Index, : ], axis=0 )
            dfProfilesNextCDSDistUnder25 = dfProfilesNextCDSDistUnder25.append( pd.DataFrame( dict(zip( currProfile.index.values, nextCDSDistanceUnder25Profile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )
            
            nextCDSDistanceOver25Profile = np.mean( nativeProfiles.loc[ nextCDSDistOver25Index, : ] - randomizedProfiles.loc[ nextCDSDistOver25Index, : ], axis=0 )
            dfProfilesNextCDSDistOver25 = dfProfilesNextCDSDistOver25.append( pd.DataFrame( dict(zip( currProfile.index.values, nextCDSDistanceOver25Profile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )

            stopCodonUGAProfile = np.mean( nativeProfiles.loc[ dfProfilesStopUGAIndex, : ] - randomizedProfiles.loc[ dfProfilesStopUGAIndex, : ], axis=0 )
            dfProfilesStopUGA = dfProfilesStopUGA.append( pd.DataFrame( dict(zip( currProfile.index.values, stopCodonUGAProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )
            
            stopCodonUAAProfile = np.mean( nativeProfiles.loc[ dfProfilesStopUAAIndex, : ] - randomizedProfiles.loc[ dfProfilesStopUAAIndex, : ], axis=0 )
            dfProfilesStopUAA = dfProfilesStopUAA.append( pd.DataFrame( dict(zip( currProfile.index.values, stopCodonUAAProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )
            
            stopCodonUAGProfile = np.mean( nativeProfiles.loc[ dfProfilesStopUAGIndex, : ] - randomizedProfiles.loc[ dfProfilesStopUAGIndex, : ], axis=0 )
            dfProfilesStopUAG = dfProfilesStopUAG.append( pd.DataFrame( dict(zip( currProfile.index.values, stopCodonUAGProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )

            if sum(operonLastIndex) >= 50:  # include species with information for at least 50 operons
                operonLastProfile = np.mean( nativeProfiles.loc[ operonLastIndex, : ] - randomizedProfiles.loc[ operonLastIndex, : ], axis=0 )
                dfProfilesOperonLast = dfProfilesOperonLast.append( pd.DataFrame( dict(zip( currProfile.index.values, operonLastProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )

                operon1stAndNotLastProfile = np.mean( nativeProfiles.loc[ operon1stAndNotLastIndex, : ] - randomizedProfiles.loc[ operon1stAndNotLastIndex, : ], axis=0 )
                dfProfilesOperon1stAndNotLast = dfProfilesOperon1stAndNotLast.append( pd.DataFrame( dict(zip( currProfile.index.values, operon1stAndNotLastProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )

                operon2ndAndNotLastProfile = np.mean( nativeProfiles.loc[ operon2ndAndNotLastIndex, : ] - randomizedProfiles.loc[ operon2ndAndNotLastIndex, : ], axis=0 )
                dfProfilesOperon2ndAndNotLast = dfProfilesOperon2ndAndNotLast.append( pd.DataFrame( dict(zip( currProfile.index.values, operon2ndAndNotLastProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )

                operon3rdAndNotLastProfile = np.mean( nativeProfiles.loc[ operon3rdAndNotLastIndex, : ] - randomizedProfiles.loc[ operon3rdAndNotLastIndex, : ], axis=0 )
                dfProfilesOperon3rdAndNotLast = dfProfilesOperon3rdAndNotLast.append( pd.DataFrame( dict(zip( currProfile.index.values, operon3rdAndNotLastProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )

                outputBDQAtables(  ( (operonProtIdIndex, 'operonProtIdIndex'), (operonGeneIdIndex, 'operonGeneIdIndex'), (operonLastIndex, "operonLastIndex"), (operon1stAndNotLastIndex, "operon1stAndNotLastIndex"), (operon2ndAndNotLastIndex, "operon2ndAndNotLastIndex"), (operon3rdAndNotLastIndex, "operon3rdAndNotLastIndex")), taxId)

            if sum(paHighIndex) >= 50:  # include species with PA information for at least 150 (3*50) genes
                paLowProfile = np.mean( nativeProfiles.loc[ paLowIndex, : ] - randomizedProfiles.loc[ paLowIndex, : ], axis=0 )
                dfProfilesPALow = dfProfilesPALow.append( pd.DataFrame( dict(zip( currProfile.index.values, paLowProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )
                paMedProfile = np.mean( nativeProfiles.loc[ paMedIndex, : ] - randomizedProfiles.loc[ paMedIndex, : ], axis=0 )
                dfProfilesPAMed = dfProfilesPAMed.append( pd.DataFrame( dict(zip( currProfile.index.values, paMedProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )
                paHighProfile = np.mean( nativeProfiles.loc[ paHighIndex, : ] - randomizedProfiles.loc[ paHighIndex, : ], axis=0 )
                dfProfilesPAHigh = dfProfilesPAHigh.append( pd.DataFrame( dict(zip( currProfile.index.values, paHighProfile)), index=pd.Index([taxId]), columns=currProfile.index.values ) )

                
                

            #
            # Create the std-dev plots
            stds = np.std( nativeProfiles - randomizedProfiles, axis=0 )
            dfStdProfiles = dfStdProfiles.append( pd.DataFrame( dict(zip( stds.index.values, stds.values )), index=pd.Index([taxId]), columns=stds.index.values ) )  # convert series (column) into row
                

    #
    # Plot all data
    #plotProfilesDists( dfProfiles.loc[:,range(-96,98,4)], xticks=range(4,49,5), xticklabels=range(-80,99,20), x0=24, highlightSpecies=(511145,224308) )
    #plotProfilesDists( dfProfiles, highlightSpecies=(511145,224308), boxplotTicks=range(-80,81,20), yrange=(-3.1,0.5) )
    plotProfilesDists( dfProfiles,
                       highlightSpecies=(), # 511145,), #(511145,224308),
                       boxplotTicks=range(-90,111,100),
                       yrange=(-4.5, 1.4),
                       xrange=(-100,100),
                       boxplotWidth=0.04,
                       colors=('#4080f2',),
                       defaultLabel="All species (N={})".format(dfProfiles.shape[0]),
                       showBoxplots=False,
                       aspect=args.aspect)


    plotProfilesDists( dfStdProfiles,
                       highlightSpecies=(), #(511145,), #(511145,224308),
                       boxplotTicks=range(-90,111,100),
                       yrange=(0.9, 7.5),
                       xrange=(-100,100),
                       boxplotWidth=0.04,
                       colors=('#4080f2',),
                       defaultLabel="All species (N={})".format(dfStdProfiles.shape[0]),
                       fileBaseName="std",
                       showBoxplots=False,
                       aspect=args.aspect)
    

    #
    # Plot data by next CDS strand
    dfProfilesNextCDSOppositeStrand = dfProfilesNextCDSOppositeStrand.assign( strand=lambda x:'opposite' )
    dfProfilesNextCDSSameStrand     = dfProfilesNextCDSSameStrand.assign(     strand=lambda x:'same' )
            
    bystrand = dfProfilesNextCDSSameStrand.append( dfProfilesNextCDSOppositeStrand )
    #plotProfilesDists( bystrand.loc[:,list(range(-90,98,10))+['strand']], category='strand', xticks=range(1,19,2), xticklabels=range(-80,98,20), x0=9, boxplotTicks=range(-90,98,10 ) )
    plotProfilesDists( bystrand,
                       category='strand',
                       boxplotTicks=range(-90,111,100),
                       yrange=(-8.1, 0.6),
                       xrange=(-100,100),
                       boxplotWidth=0.08,
                       colors=('#3060f0','#f06030'),
                       showBoxplots=False,
                       aspect=args.aspect)

    #
    # Plot data by intergenic distance
    dfProfilesNextCDSOverlapping = dfProfilesNextCDSOverlapping.assign( intergenic=lambda x:"overlapping" )
    dfProfilesNextCDSDistUnder25 = dfProfilesNextCDSDistUnder25.assign( intergenic=lambda x:"\u2264"+"25 nt" )
    dfProfilesNextCDSDistOver25  = dfProfilesNextCDSDistOver25.assign(  intergenic=lambda x:">"     +"25 nt" )

    bydistance = dfProfilesNextCDSOverlapping.append( dfProfilesNextCDSDistUnder25.append( dfProfilesNextCDSDistOver25 ))
    plotProfilesDists( bydistance,
                       category='intergenic',
                       boxplotTicks=range(-90,111,100),
                       yrange=(-5.2, 1.4),
                       xrange=(-100,100),
                       boxplotWidth=0.08,
                       colors=('#20f0da', '#20f053', '#497df4'),
                       showBoxplots=False)

    #
    # Plot data by operonic position
    dfProfilesOperonLast          = dfProfilesOperonLast.assign(          operon_pos=lambda x:"Last" )
    print(dfProfilesOperonLast.shape)
    dfProfilesOperon1stAndNotLast = dfProfilesOperon1stAndNotLast.assign( operon_pos=lambda x:"1st, not last" )
    print(dfProfilesOperon1stAndNotLast.shape)
    dfProfilesOperon2ndAndNotLast = dfProfilesOperon2ndAndNotLast.assign( operon_pos=lambda x:"2nd, not last" )
    print(dfProfilesOperon2ndAndNotLast.shape)
    dfProfilesOperon3rdAndNotLast = dfProfilesOperon3rdAndNotLast.assign( operon_pos=lambda x:"3rd, not last" )
    print(dfProfilesOperon3rdAndNotLast.shape)
    byoperonpos = dfProfilesOperonLast.append( dfProfilesOperon1stAndNotLast.append( dfProfilesOperon2ndAndNotLast.append( dfProfilesOperon3rdAndNotLast ) ))
    print(byoperonpos.shape)
    plotProfilesDists( byoperonpos,
                       category='operon_pos',
                       boxplotTicks=range(-90,111,100),
                       yrange=(-5.2, 1.4),
                       xrange=(-100,100),
                       boxplotWidth=0.08,
                       colors=('#28e85d', '#f06e20','#f81835', '#f02091'),
                       showBoxplots=False)


    dfProfilesStopUAG = dfProfilesStopUAG.assign( stop_codon=lambda x:"UAG" )
    dfProfilesStopUAA = dfProfilesStopUAA.assign( stop_codon=lambda x:"UAA" )
    dfProfilesStopUGA = dfProfilesStopUGA.assign( stop_codon=lambda x:"UGA" )
    bystopcodon = dfProfilesStopUAG.append( dfProfilesStopUAA.append( dfProfilesStopUGA ))
    plotProfilesDists( bystopcodon,
                       category='stop_codon',
                       boxplotTicks=range(-90,111,100),
                       yrange=(-5.2, 1.4),
                       xrange=(-100,100),
                       boxplotWidth=0.08,
                       colors=('#28e85d', '#f06e20','#f81835'),
                       showBoxplots=False)


    dfProfilesPALow          = dfProfilesPALow.assign(          rel_pa=lambda x:"A_Low"  )
    dfProfilesPAMed          = dfProfilesPAMed.assign(          rel_pa=lambda x:"B_Med"  )
    dfProfilesPAHigh         = dfProfilesPAHigh.assign(         rel_pa=lambda x:"C_High" )
    byrelpa = dfProfilesPALow.append( dfProfilesPAMed.append( dfProfilesPAHigh ))
    plotProfilesDists( byrelpa,
                       category='rel_pa',
                       boxplotTicks=range(-90,111,100),
                       yrange=(-4.5, 0.5),
                       xrange=(-100,100),
                       boxplotWidth=0.08,
                       colors=('#ffdd39', '#fc882b', '#ea1111'),
                       showBoxplots=False,
                       aspect=args.aspect)
    
    # byrelpa = dfProfilesPALow.append( dfProfilesPAHigh )
    # #byrelpa = byrelpa2.assign( rel_pa=byrelpa['rel_pa'] )
    # plotProfilesDists( byrelpa,
    #                    category='rel_pa',
    #                    boxplotTicks=range(-90,111,100),
    #                    yrange=(-4.5, 0.5),
    #                    xrange=(-100,100),
    #                    boxplotWidth=0.08,
    #                    colors=('#ffdd39', '#ea1111'),
    #                    showBoxplots=False,
    #                    aspect=args.aspect)
    
    plotPanelizedDLFEComparison( byoperonpos,
                                 #category='operon_pos',
                                 #boxplotTicks=range(-90,111,100),
                                 yrange=(-5.2, 1.4),
                                 xrange=(-100,100))
                                 #boxplotWidth=0.08,
                                 #colors=('#28e85d', '#f06e20','#f81835', '#f02091'),
                                 #showBoxplots=False)


def savePrunedTree(tree):
    tree.write(format=1, outfile="nmicro_s6_pruned_with_names.nw")
    
    tree = tree.copy()


    usedTaxIds = {}


    # Rewrite the tax-id into the name (so it appears in the output tree)
    # Also warn about collisions (duplicate nodes that were mapped to the tax-id)
    for node in tree.traverse():
        if not node.is_leaf():
            continue

        if node.taxId in usedTaxIds:
            print("=="*20)
            print("WARNING: Duplicate found for tax-id %d" % node.taxId)
            print("Old: %s" % usedTaxIds[node.taxId])
            print("New: %s" % node.label)
            print(node.lineageItems)
        else:
            usedTaxIds[node.taxId] = node.label # store to detect future collisions

        # Rewrite the tax-id into the name (so it appears in the output tree)
        node.name = node.taxId 
    
    tree.write(format=1, outfile="nmicro_s6_pruned_with_taxids.nw")

    
def parseList(conversion=str):
    def convert(values):
        return list(map(conversion, values.split(",")))
    return convert

def findDescendentsOfAncestor(taxIds, ancestorOfNodesToKeep):
    ret = set()
    for taxId in taxIds:
        lineage = frozenset(ncbiTaxa.get_lineage(taxId))

        if( ancestorOfNodesToKeep in lineage ):
            ret.add( taxId )
            
    return list(ret)
        

def standalone():
    import argparse
    from glob import glob
    import os

    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--use-profile-data",             type=parseList(str), default=())
    argsParser.add_argument("--use-tree",                     choices=("taxonomic", "hug", "taxonomic-collapsed", "PCA", "profile-dists" ), default="taxonomic")
    #argsParser.add_argument("--use-phylosignal-data", type=str, default="")
    argsParser.add_argument("--plot-wilcoxon",                action="store_true", default=False)
    argsParser.add_argument("--limit-taxonomy",               type=int,            default=None)
    argsParser.add_argument("--include-all-species",          action="store_true", default=False)
    argsParser.add_argument("--hide-legend",                  action="store_true", default=False)
    argsParser.add_argument("--hide-environmental-vars",      action="store_true", default=False)
    argsParser.add_argument("--profiles-Y-offset-workaround", type=float,          default=0.0)
    argsParser.add_argument("--profile-scale",                type=float,          default=1.0)
    #argsParser.add_argument("--profile-height", type=float, default=1.0)
    argsParser.add_argument("--font-size",                    type=float,          default=6)
    argsParser.add_argument("--X-server",                     action="store_true", default=False)
    argsParser.add_argument("--highlight-species",            type=parseList(int), default=())
    argsParser.add_argument("--use-Y-range",                  type=float,          default=None)
    argsParser.add_argument("--add-PCA-loading-vectors",      type=parseList(int))
    argsParser.add_argument("--PCA-loading-vectors-scale",    type=float,          default=1.0)
    argsParser.add_argument("--PCA-legend-x-pos",             type=float,          default=0.0)
    argsParser.add_argument("--zoom",                         type=float,          default=1.0)
    argsParser.add_argument("--symbol-scale",                 type=float,          default=8.0)
    argsParser.add_argument("--trait-to-plot",                type=str,            default="GC")
    argsParser.add_argument("--aspect",                       type=float,          default=0.2)
    args = argsParser.parse_args()


    global yScale
    if not args.use_Y_range is None:
        yScale = [-args.use_Y_range, args.use_Y_range]

    phylosignalProfiles = None
    #if( args.use_phylosignal_data ):
    #    phylosignalProfiles = loadPhylosignalProfiles( args.use_phylosignal_data )

    if( args.use_tree=="hug" ):
        raise Exception("Not Impl")
    #     taxa = getSpeciesToInclude()
        
    #     (completeTree, prunedTree) = pruneReferenceTree_Nmicrobiol201648(taxa) # prune complete reference phylogenetic tree to include only dataset species 
    #     if( not args.limit_taxonomy is None ):  # limit taxonomy to specified taxon
    #         prunedTree = pruneTreeByTaxonomy( prunedTree, args.limit_taxonomy  )

    #     if args.include_all_species:
    #         print("{} =ext=> ".format(len(prunedTree)))
    #         prunedTree = extendTreeWithSpecies( tree=prunedTree, additionalSpecies=taxa, limitTaxonomy=args.limit_taxonomy )
    #         print(" =ext=> {}".format(len(prunedTree)))
        
    #     drawTrees( completeTree, prunedTree, args=args )

    #     savePrunedTree( prunedTree )

    #     return 0
    
    elif( args.use_tree=="taxonomic-collapsed" ):
        raise Exception("Not Impl")
        

    #     if( not args.use_profile_data ):
    #         raise Exception()
        
    #     files = []

    #     if args.use_profile_data:
    #         files = [[x for x in glob(profilesGlob) if os.path.exists(x)] for profilesGlob in args.use_profile_data]

    #     #print(files)
        
    #     #print("Loading profile data for %d files..." % len(files))
    #     #(xdata, ydata, ydata_nativeonly, ydata_shuffledonly, labels, groups, filesUsed, biasProfiles, dfProfileCorrs, summaryStatistics) = loadProfileData(files)

    #     tileGenerator = ProfileDataCollection(files, phylosignalProfiles, externalYrange = args.use_Y_range)
        
    #     return plotCollapsedTaxonomicTree(tileGenerator, ownXserver=args.X_server)

    elif( args.use_tree=="PCA" ):
        raise Exception("Not Impl")

    #     files = []

    #     if args.use_profile_data:
    #         files = [[x for x in glob(profilesGlob) if os.path.exists(x)] for profilesGlob in args.use_profile_data]

    #     #print("Loading profile data for %d files..." % len(files))
    #     #(xdata, ydata, ydata_nativeonly, ydata_shuffledonly, labels, groups, filesUsed, biasProfiles, dfProfileCorrs, summaryStatistics) = loadProfileData(files)

    #     tileGenerator = ProfileDataCollection(files, phylosignalProfiles, externalYrange = args.use_Y_range)

    #     print("Fetching trait values for plotting...")
    #     traitValues = createTraitMapping( args.trait_to_plot )

    #     if not args.limit_taxonomy is None:
    #         filtered = findDescendentsOfAncestor(tileGenerator.getTaxIds(), args.limit_taxonomy)
    #         biasProfiles = tileGenerator.getBiasProfiles(profilesGroup=0, taxIdsToInclude = filtered)
    #     else:
    #         biasProfiles = tileGenerator.getBiasProfiles(profilesGroup=0)

    #     return PCAForProfiles( biasProfiles, tileGenerator.getYRange(), profilesYOffsetWorkaround=args.profiles_Y_offset_workaround, profileScale=args.profile_scale, fontSize=args.font_size, overlapAction="hide", highlightSpecies=args.highlight_species, addLoadingVectors=args.add_PCA_loading_vectors, loadingVectorsScale=args.PCA_loading_vectors_scale, zoom=args.zoom, legendXpos=args.PCA_legend_x_pos, traitValues=traitValues, symbolScale=args.symbol_scale )

    elif( args.use_tree=="taxonomic" ):
        files = []

        if args.use_profile_data:
            files = [[x for x in glob(profilesGlob) if os.path.exists(x)] for profilesGlob in args.use_profile_data]

        #print(files)
        
        #print("Loading profile data for %d files..." % len(files))
        #(xdata, ydata, ydata_nativeonly, ydata_shuffledonly, labels, groups, filesUsed, biasProfiles, dfProfileCorrs, summaryStatistics) = loadProfileData(files)

        #tileGenerator = ProfileDataCollection(files, externalYrange = args.use_Y_range, plotWilcoxon=False, plotWilcoxon50_100=True )
        tileGenerator = ProfileDataCollection(files, externalYrange = args.use_Y_range, plotWilcoxon=None, plotWilcoxon50_100=None )

        #return plotSpeciesOnTaxonomicTree()
        ret = plotSpeciesOnTaxonomicTree(tileFunc=tileGenerator.getProfileTileFunc(), hideLegend=args.hide_legend, hideEnvironmentalVars=args.hide_environmental_vars, ownXserver=args.X_server, treeScale=20, highlightSpecies=frozenset(args.highlight_species), numProfileGroups=tileGenerator.getNumProfileGroups(), limitTaxonomy=args.limit_taxonomy, profileData=tileGenerator )

        scaleTile = tileGenerator.getScaleTile()

        print("RTS model:")
        print( tileGenerator.getRTSStatistics() )

        tileGenerator.printStatisticts()

        print( "--"*20 )
        print( "Y scale used: {} (external: {})".format(tileGenerator.getYRange(), bool(not args.use_Y_range is None) ) )
        print( "--"*20 )
        
        return ret
    
    elif( args.use_tree=="profile-dists" ):
        files = []

        if args.use_profile_data:
            files = [[x for x in glob(profilesGlob) if os.path.exists(x)] for profilesGlob in args.use_profile_data]

        profiles = ProfileDataCollection(files, externalYrange = args.use_Y_range, plotWilcoxon=False, plotWilcoxon50_100=False, loadSequenceProfiles=True  )

        ret = plotProfilesDistribution( args, profiles )
        
        return ret


if __name__=="__main__":
    import sys
    sys.exit(standalone())


        
    
