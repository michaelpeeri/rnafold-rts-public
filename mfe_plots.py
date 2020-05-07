from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
import sys
from datetime import datetime
import numpy as np
import numpy.linalg
import pandas as pd
from math import log10, sqrt, exp
from bisect import bisect_left
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress, wilcoxon, gaussian_kde
import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
#matplotlib.use("cairo")
import matplotlib.pyplot as plt
#plt.style.use('ggplot') # Use the ggplot style
from data_helpers import getSpeciesName, getSpeciesFileName, getGenomicGCContent, getSpeciesProperty, getSpeciesShortestUniqueNamesMapping
from sklearn import decomposition
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold
import seaborn as sns
import cairo
from pyqtree import Index
from ncbi_entrez import getTaxonomicGroupForSpecies
from rate_limit import RateLimit
from mysql_rnafold import getWindowWidthForComputationTag


def plotMFEProfile(taxId, profileId, data, computationTag=None, yRange=(-12,0), showRandomizedRange=True, showComponents=True, fileLabel="", aspect=15.0, tickValues = range(-100,101,20), linewidth=2):
    #fig, (ax1,ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig, ax1 = plt.subplots()

    varsForRange = []
    
    if showComponents:
        data[['native']].plot(  color="#20b040", label="Native LFE",     linewidth=linewidth, ax=ax1, zorder=10)
        data[['shuffled']].plot(color="#20b0ff", label="Randomized LFE", linewidth=linewidth, ax=ax1, zorder=10)
        ax1.fill_between( data.index, data['native'], data['shuffled'], color="#ffb020",   alpha=0.7, zorder=1 )
        varsForRange = varsForRange + ['native', 'shuffled']
    #data['native'].plot(ax=ax1)
    #data['shuffled'].plot(ax=ax1)
    if showRandomizedRange:
        data[['shuffled75', 'shuffled25']].plot(ax=ax1, style='--', zorder=10)
        varsForRange = varsForRange + ['shuffled75', 'shuffled25']


    windowWidth = None
    if not computationTag is None:
        windowWidth = getWindowWidthForComputationTag( computationTag )

    dlfe = data['native']-data['shuffled']
    ax1.plot( data.index, dlfe, zorder=10, color="#ffb020", label=u"\u0394LFE", linewidth=linewidth )
    

    minY = min(yRange[0], np.min(np.min(data[varsForRange])), min(dlfe) )
    maxY = max(yRange[1], np.max(np.max(data[varsForRange])), max(dlfe) )
    
    
    speciesName = getSpeciesName(taxId)

    plt.title(speciesName)

    plt.xlabel('Position (nt, window start, from stop codon)')

    ax1.set_title("Mean LFE for %s" % speciesName)
    #ax1.set_ylabel(u"\u0394LFE")
    ax1.set_ylabel(u"LFE (kcal/mol/window)")
    ax1.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.4,0.0) )
    ax1.grid(axis="y")


    #data['gc'].plot(ax=ax2)
    #ax2.set_title("GC%")
    #ax2.set_ylabel('GC% (in window)')
    #ax2.grid(True)


    if not computationTag is None:
        if not fileLabel:
            baseName = "mfe_v2_40nt_cds_{}_{}_series{}".format(profileId, getSpeciesFileName(taxId), computationTag)
        else:
            baseName = "mfe_v2_40nt_cds_{}_{}_series{}_{}".format(profileId, getSpeciesFileName(taxId), computationTag, fileLabel)
    else:
        if not fileLabel:
            baseName = "mfe_v2_40nt_cds_{}_{}".format(profileId, getSpeciesFileName(taxId))
        else:
            baseName = "mfe_v2_40nt_cds_{}_{}_{}".format(profileId, getSpeciesFileName(taxId), fileLabel)
            
        
    #profileId = "tbd" # str(args.profile.ProfileId).replace(':', '-')

    ax1.set_aspect(aspect)

    if not windowWidth is None:
        ax1.annotate( "{}nt".format(windowWidth), xy=(180, minY), xytext=(180+windowWidth, minY), fontSize=10, color="black", zorder=50 )
        ax1.annotate( "".format(windowWidth),     xy=(180, minY), xytext=(180+windowWidth, minY), fontSize=6, arrowprops=dict(arrowstyle='<-', alpha=1.0, linewidth=1.0, color="black"), color="black", zorder=50 )

    ax1.set_xticks(list(tickValues))
    ax1.set_xticklabels([str(x) for x in tickValues])
    plt.xlim( (-100, 100) )
    plt.ylim( (minY-0.2, maxY+0.2) )
    plt.axvline( x=0.0, color="black", linewidth=linewidth*0.7, alpha=1.0, zorder=1 )
    plt.axhline( y=0.0, color="black", linewidth=linewidth*0.7, alpha=1.0, zorder=1 )
    
    plt.savefig("{}.pdf".format( baseName ), bbox_inches="tight", metadata={"CreationDate":datetime.utcnow().isoformat(' ')})
    plt.savefig("{}.svg".format( baseName ))
    plt.close(fig)


def plot2WayMFEComparison(taxId, profileId, data1, data2, computationTag=None, yRange=(-12,0), comment=None):
    #fig, (ax1,ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig, ax1 = plt.subplots()

    #data1[['native', 'shuffled']].plot(ax=ax1, zorder=10, style='--')
    #data2[['native', 'shuffled']].plot(ax=ax1, zorder=10)
    #data['native'].plot(ax=ax1)
    #data['shuffled'].plot(ax=ax1)
    #data[['shuffled75', 'shuffled25']].plot(ax=ax1, style='--', zorder=10)
    ax1.plot( data1.index, data1['native'],   zorder=10, label=u"Native",   linestyle='dashed', c='#e24a33' )
    ax1.plot( data1.index, data1['shuffled'], zorder=10, label=u"Shuffled", linestyle='dashed', c='#348abd' )
    ax1.plot( data2.index, data2['native'],   zorder=10, label=u"Native",   linestyle='solid',  c='#e24a33' )
    ax1.plot( data2.index, data2['shuffled'], zorder=10, label=u"Shuffled", linestyle='solid',  c='#348abd' )
    
    minY = min(yRange[0], np.min(np.min(data1[['native', 'shuffled']])))
    maxY = max(yRange[1], np.max(np.max(data1[['native', 'shuffled']])))


    windowWidth = None
    if not computationTag is None:
        windowWidth = getWindowWidthForComputationTag( computationTag )

    smfe1 = data1['native']-data1['shuffled']
    ax1.plot( data1.index, smfe1, zorder=10, label=u"\u0394LFE", linestyle='dashed', c='#fbc15e'  )
    smfe2 = data2['native']-data2['shuffled']
    ax1.plot( data2.index, smfe2, zorder=10, label=u"\u0394LFE", linestyle='solid',  c='#fbc15e'  )
    
    speciesName = getSpeciesName(taxId)

    plt.title(speciesName)

    plt.xlabel('Position (nt, window start, from stop codon)')

    ax1.set_title("Mean LFE for %s" % speciesName)
    #ax1.set_ylabel(u"\u0394LFE")
    ax1.set_ylabel(u"LFE (kcal/mol/window)")
    ax1.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.4,0.0) )
    ax1.grid(True)


    #data['gc'].plot(ax=ax2)
    #ax2.set_title("GC%")
    #ax2.set_ylabel('GC% (in window)')
    #ax2.grid(True)


    if not computationTag is None:
        baseName = "mfe_v2_40nt_cds_{}_{}_series{}_comparison".format(profileId, getSpeciesFileName(taxId), computationTag)
    else:
        baseName = "mfe_v2_40nt_cds_{}_{}_comparison".format(profileId, getSpeciesFileName(taxId))
        
    #profileId = "tbd" # str(args.profile.ProfileId).replace(':', '-')

    ax1.set_aspect(15.0)
    plt.axvline( x=0.0, color="black", linewidth=0.5, alpha=1.0, zorder=1 )

    if not windowWidth is None:
        ax1.annotate( "{}nt".format(windowWidth),    xy=(180, minY), xytext=(180+windowWidth, minY), fontSize=10, color="black", zorder=50 )
        ax1.annotate( "".format(windowWidth),    xy=(180, minY), xytext=(180+windowWidth, minY), fontSize=6, arrowprops=dict(arrowstyle='<-', alpha=1.0, linewidth=1.0, color="black"), color="black", zorder=50 )

    if comment:
        ax1.annotate( comment, xy=(-100, minY), xytext=(-100, minY), fontSize=10, color="black", zorder=50 )
        

    plt.ylim( (minY-0.2, maxY+0.2) )
    
    plt.savefig("{}.pdf".format( baseName ), bbox_inches="tight", metadata={"CreationDate":datetime.utcnow().isoformat(' ')})
    plt.savefig("{}.svg".format( baseName ))
    plt.close(fig)

def plotMultipleMFEComparison(taxId, profileId, series, computationTag=None, yRange=(-12,0), plotname="comparison", showComponents=True, showLegend=True, comment=None, xTickValues=range(-100,101,20), yTickValues=range(-7,1,1), lineWidth=2, aspect=15.0, destinations=None):
    if not destinations is None:
        assert(len(destinations)==len(series))

    fig, ax1 = plt.subplots()
    #fig, (ax1,ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    if showComponents:
        lineStyles = ['solid',   'dashed', 'dotted',  'dashdot']
        lineColors = ['#e9bb42', '#e9bb42','#e9bb42', '#e9bb42']
    else:
        lineStyles = ['solid',   'solid',   'solid',   'solid'  ]
        lineColors = ['#eb7b6d', '#6debbc', '#6dc8eb', '#6da6eb']

    minY =  1e10
    maxY = -1e10
    varsForRange = []
    
    for n, s in enumerate(series):
        data, label = s
        if showComponents:
            ax1.plot( data.index, data['native'],   zorder=10, label=u"{}, Native".format(label),   linestyle=lineStyles[n], linewidth=lineWidth, c='#e24a33' )
            ax1.plot( data.index, data['shuffled'], zorder=10, label=u"{}, Shuffled".format(label), linestyle=lineStyles[n], linewidth=lineWidth, c='#348abd' )
            #ax1.plot( data2.index, data2['native'],   zorder=10, label=u"Native",   linestyle='solid',  c='#e24a33' )
            #ax1.plot( data2.index, data2['shuffled'], zorder=10, label=u"Shuffled", linestyle='solid',  c='#348abd' )
            varsForRange = varsForRange + ['native', 'shuffled']
            
        dlfe = data['native']-data['shuffled']

        
        if not destinations is None:
            count = np.min( destinations[n].nativeMeanProfile.counts())
            label = u"{}, \u0394LFE (N={})".format(label, count)
        else:
            label = u"{}, \u0394LFE".format(label)
    
        
        ax1.plot( data.index, dlfe, zorder=10, label=label, linestyle=lineStyles[n], linewidth=lineWidth, c=lineColors[n], alpha=0.8  )
        #smfe2 = data2['native']-data2['shuffled']
        #ax1.plot( data2.index, smfe2, zorder=10, label=u"\u0394LFE", linestyle='solid',  c='#fbc15e'  )
    
        minY = min(minY, yRange[0], np.min(np.min(data[varsForRange])), min(dlfe) )
        maxY = max(maxY, yRange[1], np.max(np.max(data[varsForRange])), max(dlfe) )



    windowWidth = None
    if not computationTag is None:
        windowWidth = getWindowWidthForComputationTag( computationTag )

    
    speciesName = getSpeciesName(taxId)

    hfont = {'fontname':'Helvetica'}
    plt.title(speciesName, **hfont)


    plt.xlabel('Position (nt, window start, from stop codon)', **hfont)
    
    plt.axvline( x=0.0, color="black", linewidth=1.5, alpha=1.0, zorder=1 )
    plt.axhline( y=0.0, color="black", linewidth=1.5, alpha=1.0, zorder=1 )
    ax1.set_xticks(list(xTickValues))
    ax1.set_xticklabels([str(x) for x in xTickValues])
    ax1.set_yticks(list(yTickValues))
    ax1.set_yticklabels([str(x) for x in yTickValues])
    plt.xlim( (-100, 100) )
    plt.ylim( (minY-0.2, maxY+0.2) )

    if showComponents:
        ax1.set_ylabel(u"LFE (kcal/mol/window)", **hfont)
    else:
        ax1.set_ylabel(u"\u0394LFE (kcal/mol/window)", **hfont)

    if showLegend:
        ax1.legend(fontsize=8, loc="lower left", bbox_to_anchor=(-0.4,0.0) )
    ax1.grid(axis="y")



    #data['gc'].plot(ax=ax2)
    #ax2.set_title("GC%")
    #ax2.set_ylabel('GC% (in window)')
    #ax2.grid(True)


    if not computationTag is None:
        baseName = "mfe_v2_40nt_cds_{}_{}_series{}_{}".format(profileId, getSpeciesFileName(taxId), computationTag, plotname)
    else:
        baseName = "mfe_v2_40nt_cds_{}_{}_{}".format(profileId, getSpeciesFileName(taxId), plotname)
        
    #profileId = "tbd" # str(args.profile.ProfileId).replace(':', '-')

    ax1.set_aspect( aspect )
    #plt.axvline( x=0.0, color="black", linewidth=0.5, alpha=1.0, zorder=1 )

    if not windowWidth is None:
        ax1.annotate( "{}nt".format(windowWidth),    xy=(180, minY), xytext=(180+windowWidth, minY), fontSize=10, color="black", zorder=50 )
        ax1.annotate( "".format(windowWidth),    xy=(180, minY), xytext=(180+windowWidth, minY), fontSize=6, arrowprops=dict(arrowstyle='<-', alpha=1.0, linewidth=1.0, color="black"), color="black", zorder=50 )

    if comment:
        ax1.annotate( comment, xy=(-100, minY), xytext=(-100, minY), fontSize=10, color="black", zorder=50 )
        

    plt.ylim( (minY-0.2, maxY+0.2) )
    
    plt.savefig("{}.pdf".format( baseName ), bbox_inches="tight", metadata={"CreationDate":datetime.utcnow().isoformat(' ')})
    plt.savefig("{}.svg".format( baseName ))
    plt.close(fig)


def plot2dProfile( hist2d, hist2d_xedges, hist2d_yedges, taxId ):
    fig, ax1 = plt.subplots()
    #ax1.imshow( hist2d, cmap='bwr', aspect='auto', norm=cmapNormalizer, interpolation="bilinear" )
    ax1.imshow( np.log(hist2d) )

    # Add means line
    means = np.sum( hist2d * (hist2d_yedges[1:] + hist2d_yedges[:-1])*0.5, axis=1 ) / np.sum(hist2d, axis=1)
    plt.plot( (means+30)/60*100, range( hist2d_xedges.shape[0]-1 ), c="black" )
    
    plt.savefig( "mfe_v3_2dhist_{}.png".format( taxId ) )
    plt.close(fig)

def plot2dProfileAsRidgeplot( hist2d, hist2d_xedges, hist2d_yedges, taxId ):
    #a = pd.DataFrame(np.zeros((199,100)), index=range(-100,99), columns=list(range(-50,50)))
    f, ax = plt.subplots()
    sns.set(style="white", rc={"axes.facecolor":(0,0,0,0)})
    
    a = pd.DataFrame(hist2d, index=hist2d_xedges[:-1], columns=hist2d_yedges[:-1] )
    a = a.iloc[range(0,200,3), :]
    df = a.melt(value_vars=hist2d_yedges[:-1])
    #df["pos"] = pd.Series( np.tile( np.round(np.array(hist2d_xedges[range(76,125,8)])), 100 ), dtype=np.int )
    #np.tile( np.array(np.round(hist2d_xedges[range(76,125,8)]), np.int), 3)
    #df["pos"] = pd.Categorical( np.tile( list("ABCDEFG"), 100 ), categories=list("ABCDEFG"), ordered=True)
    allPositions = np.array( np.round( hist2d_xedges[range(0,200,3)]), dtype=np.int)
    df["pos"] = pd.Series( np.tile( allPositions, 100 ) )
    #df["xxyx"] = pd.Series( 
    pal = sns.cubehelix_palette(len(range(0,200,3)), rot=-.25, light=.7 )
    #g = sns.FacetGrid(df, col="pos", hue="pos", aspect=0.05, height=5.0, palette=pal, sharex=False, sharey=True, col_order=list(reversed(hist2d_xedges[range(0,200,3)])) )
    g = sns.FacetGrid(df, col="pos", hue="pos", aspect=0.05, height=5.0, palette=pal, sharex=False, sharey=True, col_order=list(reversed(allPositions)) )
    #g.map( sns.lineplot, x="variable", y="value", data=df )
    g.map( plt.fill_between, "value", "variable", data=df )
    g.map( plt.plot,         "value", "variable", color="white", zorder=10, linewidth=1 )
    #g.map( sns.lineplot, "value", "variable", color="white", lw=2, data=df )

    #def test(x, y, color, label):
    #    print(x, y)
    #g.map(test, "variable", "value")

    #g.map( sns.kdeplot, "value", clip_on=False, shade=True )
    #g.map( plt.set_yscale, "log" )
    g.despine( bottom=True, left=True )
    g.fig.subplots_adjust(wspace=-.9)
    
    plt.savefig("ridge_{}.pdf".format(taxId))
    plt.savefig("ridge_{}.svg".format(taxId))

    # for i in range(8):
    #     f, ax = plt.subplots()
    #     plt.plot( a.iloc[i,:] )
        
    #     plt.title(i)
    #     plt.savefig("ridge_{}_debug_{}".format(taxId, i))

    f, ax = plt.subplots()
    for n, x in enumerate(a.index):
        plt.plot( a.iloc[n,:], c="green")
        dd = df[df["pos"]==x]
        plt.plot( dd["variable"], dd["value"], c="blue")
    plt.savefig("ridge_{}_debug.pdf".format(taxId))
        

class RGBinterpolator(object):
    def __init__(self, N, c=[[0.79, 0.01, 0.01], [0.03, 0.26, 0.99], [0.6, 0.94, 0.55]] ):
        self.c = c
        assert(len(c)==3)
        self.delta = [[c[1][0]-c[0][0], c[1][1]-c[0][1], c[1][2]-c[0][2]], [c[2][0]-c[1][0], c[2][1]-c[1][1], c[2][2]-c[1][2]]]
        self.N = N

    def getColor(self, n):
        pos = float(n)/(self.N-1)
        if pos < 0.5:
            return (self.c[0][0] + self.delta[0][0]*((pos-0.0)/0.5),
                    self.c[0][1] + self.delta[0][1]*((pos-0.0)/0.5),
                    self.c[0][2] + self.delta[0][2]*((pos-0.0)/0.5))
        else:
            return (self.c[1][0] + self.delta[1][0]*((pos-0.5)/0.5),
                    self.c[1][1] + self.delta[1][1]*((pos-0.5)/0.5),
                    self.c[1][2] + self.delta[1][2]*((pos-0.5)/0.5))
        
def plotMultipleDestinations(taxId, plotLabel, profiles, ranges, xpositions):
    assert(len(ranges)==len(profiles))
    f, ax = plt.subplots()

    clr = RGBinterpolator(N=len(profiles))
    labels = []
    for i, profile in enumerate(profiles):
        r = ranges[i]
        dlfe = profile.nativeMeanProfile.value() - profile.shuffledMeanProfile.value()
        ax.plot( xpositions, dlfe, c=clr.getColor(i), label=profile.subclass  )
        y = -5.8 + i*0.2
        ax.annotate( "", xy=(0,y), xytext=(0.5*(r[0]+r[1]),y), color=clr.getColor(i), arrowprops=dict(arrowstyle='-', alpha=1.0, linewidth=1.5, color=clr.getColor(i)) )
        ax.annotate( "{}-{}".format(r[0],r[1]), xy=(min(95, 0.5*(r[0]+r[1])),y), color=clr.getColor(i) )
        labels.append("{}-{}".format(r[0],r[1]))
    ax.legend(labels)

    plt.axvline( x=0.0, color="black", linewidth=0.5, alpha=1.0, zorder=1 )
    plt.axhline( y=0.0, color="black", linewidth=0.5, alpha=1.0, zorder=1 )

    plt.xlabel( "Distance from stop codon (nt)" )
    plt.ylabel( "dLFE" )

    speciesName = getSpeciesName(taxId)
    plt.title(speciesName)
    
    plt.savefig("{}_{}.pdf".format(plotLabel, taxId))
    plt.savefig("{}_{}.svg".format(plotLabel, taxId))

def plotMFEProfileV3(taxid, profileId, df, dLFEData, wilcoxon, transitionPeak, transitionPeakPos, edgeWilcoxon, ProfilesCount):
    # TODO - RE-IMPL THIS
    pass

#         spearman_smfe_CAI_pval  spearman_smfe_CAI_rho          ...           spearman_smfe_gc_pval  spearman_smfe_gc_rho

def plotSmoothedLengthEffectData(taxid, plotLabel, df, xvar='Pos'):
    f, ax = plt.subplots()
    df.plot(x=xvar, y='MinDLFE', color="blue",              label="MinDLFE", ax=ax)
    df.plot(x=xvar, y='Mean',    color="green",             label="Mean",    ax=ax)
    df.plot(x=xvar, y='P25',     color="grey",  style='--', label="P25",     ax=ax)
    df.plot(x=xvar, y='P75',     color="grey",  style='--', label="P75",     ax=ax)
    df.plot(x=xvar, y='Density', color="black", style='-',  label="Density", secondary_y=True, ax=ax)
    speciesName = getSpeciesName(taxid)
    plt.title(speciesName)
    plt.savefig("{}_{}.pdf".format(plotLabel, taxid))
    plt.savefig("{}_{}.svg".format(plotLabel, taxid))

def plotMFEvsCUBcorrelation( biasProfiles, cubCorrs ):

    yvars = ('spearman_smfe_Nc_rho', 'spearman_smfe_CAI_rho', 'spearman_smfe_Fop_rho', 'spearman_smfe_gc_rho')
    
    print(cubCorrs.columns)
    for cdsPos in (0, 25):
        #xvals = [x.iat[cdsPos] for x in biasProfiles.values()]
        #print(xvar)

        for yvar in yvars:
            yvals = cubCorrs[yvar].values
            yvarWords = yvar.split("_")

            xvals = []
            for taxid in cubCorrs.index.values:
                xvals.append( biasProfiles[taxid].iat[cdsPos] )
            #print(xvals)
            #print(yvals)

            f, ax = plt.subplots()
            g = sns.jointplot( xvals, yvals, kind="scatter", dropna=False )
            plt.xlabel("dLFE at begin+{}nt".format(cdsPos*10))
            plt.ylabel("Spearman correlation vs. {}".format(yvarWords[2]))
            plt.ylim((-1,1))
            g.savefig("cub_corrs_{}_vs_dLFE_{}_begin.pdf".format(yvar, cdsPos*10) )
            g.savefig("cub_corrs_{}_vs_dLFE_{}_begin.svg".format(yvar, cdsPos*10) )

    f, ax = plt.subplots(4,1, figsize=(7,7), sharex=True)
    ymax = 64
    for i, yvar in enumerate(yvars):
        yvals = cubCorrs[yvar].values
        yvarWords = yvar.split("_")
        yvarName = yvarWords[2]
        if yvarName=="gc": yvarName="GC%"

        g = sns.distplot(yvals, kde=False, bins=10, ax=ax[i])

        meanVal   = np.mean(yvals)

        # Show mean value
        ax[i].axvline( x=meanVal, c="red" )
        ax[i].annotate(s="mean={:.2}".format(meanVal), xy=(meanVal-0.015, ymax*0.9),  color="red", fontsize=11, horizontalalignment="right" )
        # Show x-axis (no correlation)
        ax[i].axvline( x=0, c="black" )
        # Update axis labels
        ax[i].set_xlabel("Spearman correlation vs. {}".format(yvarName))
        ax[i].set_ylabel("Num. species")
        # Used fixed y-range
        ax[i].set_ylim((0, ymax))
        
    # Annotate number of species
    ax[0].annotate(s="N={}".format(cubCorrs.shape[0]), xy=(0.82, ymax*0.9),  fontsize=11 )
                


    plt.tight_layout()
    plt.savefig("cub_corrs_all_hist_only.pdf")
    plt.savefig("cub_corrs_all_hist_only.svg")
    

        

    
randomizationTypesLabels = {11:"Codon shuffle", 12:"Vertical shuffle"}  # TODO - move this to a good place (resolve dependency problem)
def plotMFEProfileForMultipleRandomizations(taxId, profileId, data):
    fig, ax1 = plt.subplots()

    arbitraryKey = list(data.keys())[0]
    data[arbitraryKey][['native']].plot(ax=ax1)
    labels = []
    labels.append("Native")
    
    for shuffleType in list(data.keys()):
        data[shuffleType][['shuffled']].plot(ax=ax1)
        labels.append(randomizationTypesLabels[shuffleType])

    ax1.legend(labels)
    #L = plt.legend()
    #for i, label in enumerate(labels):
    #    L.get_texts()[i].set_text(label)
        
    #data[arbitraryKey][['shuffled75', 'shuffled25']].plot(ax=ax1, style='--')

    speciesName = getSpeciesName(taxId)

    plt.title(speciesName)

    plt.xlabel('Position (nt, window start, from cds start)')

    #tile = getProfileHeatmapTile(99991320, data[11], (-2.9,2.9) )
    #print(tile)
    #tileData = plt.imread(tile, format='png')
    #
    #cmapNormalizer        = CenterPreservingNormlizer(-2.9, 2.9)
    #plt.imshow( np.array( tileData ).reshape(1,-1), cmap='bwr', norm=cmapNormalizer, extent=(1400,1900,-8,-7.5), interpolation='bilinear' )
    #plt.imshow( tileData, cmap='bwr', norm=cmapNormalizer, extent=(1400,1900,-8,-7.5), interpolation='bilinear' )
    

    ax1.set_title("Mean LFE for %s" % speciesName)
    ax1.set_ylabel('Mean LFE')
    #ax1.legend(fontsize=8)
    ax1.grid(True)

    #profileId = "tbd" # str(args.profile.ProfileId).replace(':', '-')
    plt.savefig("mfe_v2_40nt_cds_%s_allshuffles_%s.pdf" % (profileId, getSpeciesFileName(taxId)) )
    plt.savefig("mfe_v2_40nt_cds_%s_allshuffles_%s.svg" % (profileId, getSpeciesFileName(taxId)) )
    plt.close(fig)

    
def plotMFEProfileMultiple(taxId, profileId, data, additionalVars, scaleBar=None):
    fig, axes = plt.subplots(2+len(additionalVars), sharex=True)

    data[['native', 'shuffled']].plot(ax=axes[0])
    #data[['shuffled75', 'shuffled25']].plot(ax=axes[0], style='--')

    smfe = data['native']-data['shuffled']
    axes[1].plot([min(data.index), max(data.index)], [0,0], c='black')
    axes[1].plot( data.index, smfe, zorder=10 )
    axes[1].set_ylabel(u"\u0394MFE")

    
    speciesName = getSpeciesName(taxId)

    plt.xlabel('Position (nt, window start, from cds start)')

    axes[0].set_title("Biases for %s" % speciesName)
    axes[0].set_ylabel(r'MFE')
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    plotRange = []
    for i, var in enumerate(additionalVars):
        currentAxis = i+2
        data[var].plot(ax=axes[currentAxis])
        #axes[currentAxis].set_title(var)
        axes[currentAxis].set_ylabel(var)
        axes[currentAxis].grid(True)

        plotRange = [min(data[var]), max(data[var])]

        yrange = (min(data[var]), max(data[var]))
        warn = ''
        if( len(data[var]) < 500 ):
            warn = ' <!>'

        spearman = spearmanr( smfe, data[var])
        axes[currentAxis].annotate(s="Signed Spearman r: %1.3f (p<%1.2f%s)" % (spearman.correlation, spearman.pvalue, warn),  xy=(max(data.index)*0.8, yrange[0]+(yrange[1]-yrange[0])*0.20),  fontsize=6 )

        spearman2 = spearmanr( abs(smfe), data[var])
        axes[currentAxis].annotate(s="Unsigned Spearman r: %1.3f (p<%1.2f%s)" % (spearman2.correlation, spearman2.pvalue, warn),  xy=(max(data.index)*0.8, yrange[0]+(yrange[1]-yrange[0])*0.05),  fontsize=6 )

    if( not scaleBar is None ):
        # Draw a scale-bar
        scaleBarPosY   = plotRange[0] + 0.75*(plotRange[1]-plotRange[0])  # use the range of the last plot
        scaleBarHeight =               0.075*(plotRange[1]-plotRange[0])  # half-height actually...
        
    
        axes[-1].plot([ 5, 5+scaleBar],          [scaleBarPosY, scaleBarPosY], c='black')
        axes[-1].plot([ 5, 5],                   [scaleBarPosY-scaleBarHeight, scaleBarPosY+scaleBarHeight], c='black')
        axes[-1].plot([ 5+scaleBar, 5+scaleBar], [scaleBarPosY-scaleBarHeight, scaleBarPosY+scaleBarHeight], c='black')

        

    #profileId = "tbd" # str(args.profile.ProfileId).replace(':', '-')
    plt.savefig("mfe_v2_40nt_cds_vars_%s_%s.pdf" % (profileId, getSpeciesFileName(taxId)) )
    plt.savefig("mfe_v2_40nt_cds_vars_%s_%s.svg" % (profileId, getSpeciesFileName(taxId)) )
    plt.close(fig)

    

def plotXY(taxId, profileId, data, xvar, yvar, title, computationTag=None):
    fig, ax1 = plt.subplots()

    data[[yvar]].plot(ax=ax1)


    #plt.annotate(s="n= %d"  % len(yvals),                                                 xy=(0.28, 17.0),  fontsize=6 )


    #plt.xlim([.25,.75])
    plt.xlabel(xvar.replace('_',' '))
    plt.ylabel(yvar.replace('_',' '))
    plt.title( title )
    plt.grid(True)
    #plt.legend(loc=(0,1), scatterpoints=1, ncol=3, fontsize='small')

    if not computationTag is None:
        baseName = "mfe_v2_40nt_%s_vs_%s_%s_%s_series%d" % (yvar, xvar, profileId, getSpeciesFileName(taxId), computationTag)
    else:
        baseName = "mfe_v2_40nt_%s_vs_%s_%s_%s" % (yvar, xvar, profileId, getSpeciesFileName(taxId))
    
    plt.savefig("{}.pdf".format(baseName))
    plt.savefig("{}.svg".format(baseName))
    plt.close(fig)
    
    
def scatterPlot(taxId, profileId, data, xvar, yvar, title="%s", colorvar=None, label="genelevel"):
    data = data.copy()
    data.dropna(subset=(xvar, yvar), inplace=True)

    fig, ax1 = plt.subplots()
    if colorvar is None:
        data.plot(x=xvar, y=yvar, ax=ax1, kind='scatter')
    else:
        data.plot(x=xvar, y=yvar, c=colorvar, s=3.0, ax=ax1, kind='scatter')



    ################################

    xvals = data[xvar]
    yvals = data[yvar]
    #print(data.head())

    # Linear correlation and factors
    pearson = pearsonr(xvals, yvals)
    spearman = spearmanr(xvals, yvals)
    kendall = kendalltau(xvals, yvals)
    l = linregress(xvals, yvals)

    abline_x = np.arange(min(xvals), max(xvals)*1.1, (max(xvals)-min(xvals)) )
    abline_y = abline_x * l.slope + l.intercept
    plt.plot(abline_x, abline_y, '--')

    topr = max(yvals)*1.05
    left = min(xvals)
    scaler = topr / 20
    # plot the linear approximation
    plt.annotate(s="Pearson $\\rho$: %1.3f (p<%g)"  % (pearson[0], pearson[1]),                 xy=(left, topr-scaler*1),  fontsize=6 )
    plt.annotate(s="Pearson $r^2$: %1.3f"  % (pearson[0]**2,),                              xy=(left, topr-scaler*2),  fontsize=6 )
    plt.annotate(s="Spearman r: %1.3f (p<%g)" % (spearman.correlation, spearman.pvalue),  xy=(left, topr-scaler*3),  fontsize=6 )
    plt.annotate(s="Kendall's tau: %1.3f (p<%g)" % (kendall.correlation, kendall.pvalue), xy=(left, topr-scaler*4),  fontsize=6 )
    plt.annotate(s="n= %d"  % len(yvals),                                                 xy=(left, topr-scaler*5),  fontsize=6 )





    ################################3


    

    #plt.annotate(s="n= %d"  % len(data),                                                 xy=(0.3, 4),  fontsize=6 )

    #plt.xlim([-200,1000])
        

    #plt.xlim([.25,.75])
    plt.xlabel(xvar.replace('_',' '))
    plt.ylabel(yvar.replace('_',' '))
    plt.title( title % getSpeciesName(taxId) )
    plt.grid(True)
    #plt.legend(loc=(0,1), scatterpoints=1, ncol=3, fontsize='small')

    plt.savefig("mfe_v2_40nt_%s_%s_vs_%s_%s_%s.pdf" % (label, yvar, xvar, profileId, getSpeciesFileName(taxId)))
    plt.savefig("mfe_v2_40nt_%s_%s_vs_%s_%s_%s.svg" % (label, yvar, xvar, profileId, getSpeciesFileName(taxId)))
    plt.close(fig)

class CenterPreservingNormlizer(matplotlib.colors.Normalize):
    def __init__(self, negativeRange, positiveRange):
        self._negativeRange = negativeRange
        assert(positiveRange>0.0)
        self._positiveRange = positiveRange
        assert(negativeRange<0.0)


        # Declare to the rest of the API what is our output range
        self.vmin=0
        self.vmax=1

    #def __call__(self, values, clip=None):
    def __call__(self, values):
        outHalfScale = 0.5*(float(self.vmax)-float(self.vmin))
        outCenter = self.vmin + outHalfScale

        out = values.copy()

        
        factor = self._positiveRange / outHalfScale
        #print("+factor: %g" % (factor))
        values[values > 0.0] /= factor
        factor = self._negativeRange / outHalfScale*-1
        #print("-factor: %g" % (factor))
        values[values <= 0.0] /= factor

        values += outCenter

        values = 1-values

        #assert(np.all(values >= self.vmin))
        #assert(np.all(values <= self.vmax))
        
        # Use the logistic function (https://en.wikipedia.org/wiki/Logistic_function) to increase 'contrast'
        # To test using gnuplot:
        # plot [0:1] 1/(1+exp(-15*(x-0.5)))
        #
        steepness = 15.0
        return 1 / (1+np.exp(-steepness*(values-0.5)))


"""
Plot the profile for a single species (contained in 'data'), save into a file, and return its name.
 taxId   - taxId of the species to plot
 data    - profile data for multiple species (including the one specified by taxId...)
 yrange  - y-range scale (this allows all tiles to have the same scale)
"""
def getProfileHeatmapTile(taxId, data, yrange, ticks=False, profileStep=10, phylosignalProfiles=None, profilesGroup=0, profileTilesFirstLineFix=False, wilcoxonDLFEZero=None, wilcoxonDLFE50_100=None):
    if not taxId in data:
        print("getProfileHeatmapTile(): taxId {} not found".format(taxId))
        return None

    assert(len(yrange)==2)
    assert(yrange[0] < yrange[1])

    isSecondaryPlotActive = (((not wilcoxonDLFEZero   is None) and (taxId in wilcoxonDLFEZero  )) or
                             ((not wilcoxonDLFE50_100 is None) and (taxId in wilcoxonDLFE50_100)))
    
    #fig, ax = plt.subplots()
    if isSecondaryPlotActive:
        fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots()
        ax2 = None

    # read profile data
    series = data[taxId]

    if profileTilesFirstLineFix: # fix for some end-profiles with invalid first element (i.e. based on very partial data)
        series = series[1:]
    
    cmapNormalizer        = CenterPreservingNormlizer(yrange[0], yrange[1])
    phylosignalNormalizer = None
    if not phylosignalProfiles is None:
        phylosignalNormalizer = CenterPreservingNormlizer(np.min(phylosignalProfiles.values), np.max(phylosignalProfiles.values) )

    #pvalNormalizer = None
    #if not (wilcoxonDLFEZero is None and wilcoxonDLFE50_100 is None):
    #    pvalNormalizer = matplotlib.colors.Normalize(vmin=-10,vmax=-1.5)
    pvalNormalizer = None
    if isSecondaryPlotActive:
        pvalNormalizer = CenterPreservingNormlizer( -50, 50 )

    # read phylosignal profile (if used)
    phylosignalProfile = None
    if( not phylosignalProfiles is None and taxId in phylosignalProfiles.index ): # note: this does not support end-referenced profiles
        phylosignalProfile = phylosignalProfiles.loc[taxId,:]
        
        if( len(series) < len(phylosignalProfile) ):
            phylosignalProfile = phylosignalProfile[:len(series)]

        assert(phylosignalProfile.index[0] == "Profile.1")
        assert(len(phylosignalProfile)==len(series))
        #print( "Got phylosignal for {}".format(taxId) )

    wilcoxonDLFEZeroProfile = None    
    #if( not wilcoxonDLFEZero is None and taxId in wilcoxonDLFEZero ):
    #    wilcoxonDLFEZeroProfile = wilcoxonDLFEZero[taxId]
    if isSecondaryPlotActive:
        wilcoxonDLFEZeroProfile = wilcoxonDLFE50_100[taxId]

        
    imdata = np.array(series).reshape(1,-1)

    #ax.axis(xmin=series.index[0], xmax=series.index[-1])

    ax1.imshow( imdata, cmap='bwr', aspect='auto', norm=cmapNormalizer, interpolation="bilinear" )

    # if( not phylosignalProfile is None ):
    #     ax2.imshow( phylosignalProfile.values.reshape(1,-1), cmap='bwr', aspect='auto', norm=phylosignalNormalizer, interpolation="bilinear" )

    if isSecondaryPlotActive:
        assert( not wilcoxonDLFEZeroProfile is None )
        # test values (legend)
        #valuesForPlotting = ( np.log10( np.maximum( 1e-320, np.concatenate( (np.linspace( 0, 1, 100), np.linspace( 1, 0, 100)) ))) * np.repeat( (-1,1), 100 )  ).reshape(1,-1)
        valuesForPlotting = ( np.log10( np.maximum( 1e-320, wilcoxonDLFEZeroProfile['pval'].values ) ) * wilcoxonDLFEZeroProfile['sign'].values * -1 ).reshape(1,-1)
        ax2.imshow( valuesForPlotting, cmap='bwr', aspect='auto', norm=pvalNormalizer, interpolation="bilinear" )
        
    #else:  # if phylosignal is not shown, the profile will be also be plotted on the 2nd axis 
    #    ax2.imshow( imdata, cmap='bwr', aspect='auto', norm=cmapNormalizer, interpolation="bilinear" )


    #pos = list(ax1.get_position().bounds)
    #taxname = getSpeciesFileName(taxId)
    #taxDescriptor = "%s.%s" % (taxname[0], taxname[1:9])
    #fig.text(pos[0]-0.01, pos[1]+pos[3]/2., taxDescriptor, va='center', ha='right', fontsize=8)

    ax1.set_yticks(())
    
    if not ax2 is None:
        ax2.set_yticks(())
        
    if ticks:
        tickValues = list(range(10, len(series)-10, 10))
        ax1.set_xticks(tickValues)
        ax1.set_xticklabels(["" for x in tickValues])
    else:
        ax1.set_xticks(())
            

    if profilesGroup == 0:
        tileFilename = "heatmap_profile_taxid_{}.png".format(taxId)
    else:
        tileFilename = "heatmap_profile_taxid_{}_g{}.png".format(taxId, profilesGroup)
    plt.savefig(tileFilename, orientation='portrait', bbox_inches='tight')
    #plt.savefig("heatmap_profile_taxid_%d.svg" % taxId, orientation='portrait')
    plt.close(fig)

    return tileFilename

"""
Plot the profile for a single species (contained in 'data'), save into a file, and return its name.
 taxId   - taxId of the species to plot
 data    - profile data for multiple species (including the one specified by taxId...)
 yrange  - y-range scale (this allows all tiles to have the same scale)
"""
def getLegendHeatmapTile(yrange):
    assert(len(yrange) == 2)
    assert(yrange[0] <= yrange[1])
    
    fig, ax = plt.subplots()

    b = 1.2

    series = np.linspace( b**yrange[0], 1 - b**(-yrange[1]), 100)  # Create a range whose logit image will cover the range yrange...
    print(series)
    #series = np.linspace( yrange[0], yrange[1], 100)   # linear scale
    series = np.log( series / (1-series)) / np.log(b)  # Logit function (inverse of logistic function)
    #print(series)
    cmapNormalizer = CenterPreservingNormlizer(yrange[0], yrange[1])

    imdata = series
    imdata = np.vstack((imdata,imdata))  # pretty crude trick borrowed from matplotlib examples...

    ax.imshow( imdata, cmap='bwr', aspect=2.0, norm=cmapNormalizer, interpolation="bilinear" )

    def roundTowardZero(x):
        if x < -0.5:
            return round(x)+1
        elif x > 0.5:
            return round(x)-1
        else:
            return 0

    #ax.set_title(taxId)
    ax.set_yticks(())
    #ax.axis(xmin=yrange[0], xmax=yrange[1])
    tick_values = list(sorted(list(range(int(roundTowardZero(yrange[0])), int(roundTowardZero(yrange[1]))+1, 2 )) + [-1.0, 1.0]))  # Put ticks at integer intervals, plus two special ticks at -.5 and .5 (since this regions spans much of the graph)
    #print(tick_values)
    tick_positions = [bisect_left(series, x) for x in tick_values]  # use bisect to find the approximate position of each tick
    #print(tick_positions)
    
    ax.set_xticks( tick_positions ) # set tick positions
    ax.set_xticklabels( ["%.2g" % x for x in tick_values], size="xx-large" )  # set tick labels

    tileFilename = "heatmap_profile_legend"
    plt.savefig("{}.svg".format(tileFilename), orientation='landscape', bbox_inches='tight' )
    plt.savefig("{}.png".format(tileFilename), orientation='landscape', bbox_inches='tight', dpi=1000)
    plt.close(fig)

    return tileFilename


def getNodeDiversityPlot(clusterRadius, groupRadius, identifier):
    fig, ax = plt.subplots()

    scale=50.0
    shapeScale=400.0

    ax.scatter(0, 0, s=clusterRadius*scale*shapeScale, facecolors="none", edgecolors="black" )
    ax.scatter(0, 0, s=groupRadius*scale*shapeScale, facecolors="none", edgecolors="blue" )

    ax.set_aspect("equal")
    ax.set_xticks( [] )
    ax.set_yticks( [] )
    ax.set_xlim((-scale*0.2, scale*0.2))
    ax.set_ylim((-scale*0.2, scale*0.2))

    tileName = "node_diversity_{}.png".format(identifier)
    plt.savefig( tileName, bbox_inches='tight', dpi=100)
    plt.close(fig)
    return tileName

def toStrWithUnicodeMinusSign(x):
    if x >= 0.0:
        return "{}".format(x)
    else:
        return "\u2212{}".format(-x)

speciesColors = ['#0ac22c', '#86dd52']

def brighten(hexcolor):
    assert(len(hexcolor)==7)
    assert(hexcolor[0]=="#")
    sqrt_brighten = lambda x: sqrt(x)
    from_hex = lambda b: int(b,16)/255
    to_hex = lambda f: hex(round(f*255))[2:]
    r = sqrt_brighten(from_hex(hexcolor[1:3]))
    g = sqrt_brighten(from_hex(hexcolor[3:5]))
    b = sqrt_brighten(from_hex(hexcolor[5:7]))
    return "#{}{}{}".format( to_hex(r), to_hex(g), to_hex(b) )

def plotProfilesDists( profiles, category=None, xticks=None, xticklabels=None, highlightSpecies=(), boxplotTicks=None, yrange=(-4,1), boxplotWidth=0.05, xrange=(-100,100), colors=('#2040f0',), defaultLabel="All", showBoxplots=True, fileBaseName=None, aspect=0.2 ):
    sns.set(style="ticks")
    assert(len(yrange)==2)
    assert(yrange[1] > yrange[0])

    fig, ax = plt.subplots()

    xvals = profiles.iloc[0,:].index
    if not category is None:
        assert(xvals[-1]==category)
        xvals = xvals[:-1]
    #transformToFigureXscale = lambda x: (x-100)/20+10
    transformToFigureXscale = lambda x: (x-190)/100+3.8
    xvals2 = [transformToFigureXscale(x) for x in xvals]

    legendLines = []
    legendCaptions = []

    pointsForKde = pd.DataFrame()

    
    def plotCategory( profiles, plotIQR=True, plotOutliers=True, color='#2030f0', boxplotXoffset=0.0, boxplotWidth=boxplotWidth, label="category"  ):
        nonlocal pointsForKde
        #ax.set_aspect(6.0)
        #p =  sns.boxplot( data=profiles.melt(id_vars=()), x='variable', y='value', color='#2398ad', fliersize=2.5, ax=ax )
        # bp = ax.boxplot( profiles.loc[:,boxplotTicks],
        #                   positions=list(boxplotTicks),
        #                   widths=0.2,
        #                   flierprops=dict(markersize=2),
        #                   patch_artist=True )

        medians = profiles.median(        axis=0)
        q25     = profiles.quantile(0.25, axis=0)
        q75     = profiles.quantile(0.75, axis=0)
        iqr     = q75-q25
        #print(profiles.shape)
        #print(q25.shape)
        #print(q75.shape)
        #print(iqr.shape)
        ocolor = brighten(color)
        
        if profiles.shape[-1] == q25.shape[-1]+1:
            q1iqr   = np.maximum( (q25 - 1.5*iqr), np.amin(profiles, axis=0)[:-1] )
            q3iqr   = np.minimum( (q75 + 1.5*iqr), np.amax(profiles, axis=0)[:-1] )
        else:
            assert(profiles.shape[-1] == q25.shape[-1])
            q1iqr   = np.maximum( (q25 - 1.5*iqr), np.amin(profiles, axis=0) )
            q3iqr   = np.minimum( (q75 + 1.5*iqr), np.amax(profiles, axis=0) )
            
        ax.plot( xvals2,      medians, color='#000000', zorder=1 )
        if profiles.shape[-1] == q25.shape[-1]+1:
            #x.fill_between( xvals2[:-1], q1iqr, q3iqr, color='white', alpha=1.0, zorder=1 )
            #ax.fill_between( xvals2[:-1], q3iqr, q1iqr, color='white', alpha=1.0, zorder=1 )
            ax.fill_between( xvals2[:-1], q25,   q75,   color=color,   alpha=1.0, zorder=1 )
            ax.fill_between( xvals2[:-1], q25,   q75,   color=color,   alpha=0.7, zorder=2 )
            if plotIQR:
                ax.plot( xvals2[:-1], q1iqr,   color=ocolor, zorder=2 )
                ax.plot( xvals2[:-1], q3iqr,   color=ocolor, zorder=2 )
        else:
            assert(profiles.shape[-1] == q25.shape[-1])
            #ax.fill_between( xvals2, q1iqr, q3iqr, color='white', alpha=1.0, zorder=1 )
            #ax.fill_between( xvals2, q3iqr, q1iqr, color='white', alpha=1.0, zorder=1 )
            ax.fill_between( xvals2, q25,   q75,   color=color,   alpha=1.0, zorder=1 )
            ax.fill_between( xvals2, q25,   q75,   color=color,   alpha=0.7, zorder=2 )
            if plotIQR:
                ax.plot( xvals2, q1iqr,   color=ocolor, zorder=2 )
                ax.plot( xvals2, q3iqr,   color=ocolor, zorder=2 )
            

        #
        # Plot outliers (for this category only)
        allPoints = profiles.melt()
        allPoints['value'] = allPoints['value'].astype('float')
        allPoints = pd.merge( allPoints, pd.DataFrame({'q1iqr':q1iqr}), how='inner', left_on='variable', right_index=True )
        allPoints = pd.merge( allPoints, pd.DataFrame({'q3iqr':q3iqr}), how='inner', left_on='variable', right_index=True )
        allPoints['figx'] = allPoints['variable'].transform( transformToFigureXscale )

        if plotOutliers:
            # plot negative outliers
            od1 = allPoints[ allPoints['value'] <= allPoints['q1iqr'] ]
            od1['fixedy'] = od1['value'].transform( lambda y: yrange[0]+0.1 )
            if not od1[ od1['value'] >= yrange[0]+0.1 ].empty:
                od1[    od1['value'] >= yrange[0]+0.1 ].plot.scatter( x='figx', y='value',  s=6.0,  color=ocolor, alpha=0.6, marker="D", zorder=2, ax=ax )
            if not od1[ od1['value'] <  yrange[0]+0.1 ].empty:
                od1[    od1['value'] <  yrange[0]+0.1 ].plot.scatter( x='figx', y='fixedy', s=13.0, color=ocolor, alpha=0.6, marker="v", zorder=2, ax=ax )

            od2 = allPoints[ allPoints['value'] >= allPoints['q3iqr'] ]
            od2['fixedy'] = od2['value'].transform( lambda y: yrange[1]-0.1 )
            if not od2[ od2['value'] <=  yrange[1]-0.1 ].empty:
                od2[    od2['value'] <=  yrange[1]-0.1 ].plot.scatter( x='figx', y='value',  s=6.0,  color=ocolor, alpha=0.6, marker="D", zorder=2, ax=ax )
            if not od2[ od2['value'] >   yrange[1]-0.1 ].empty:
                od2[    od2['value'] >   yrange[1]-0.1 ].plot.scatter( x='figx', y='fixedy', s=13.0, color=ocolor, alpha=0.6, marker="^", zorder=2, ax=ax )
        # bracket points lying outside viewable area
        allPoints.loc[ allPoints['value'] > yrange[1]-0.1 , 'value' ] = yrange[1]-0.1
        allPoints.loc[ allPoints['value'] < yrange[0]+0.1 , 'value' ] = yrange[0]+0.1
        pointsForKde = pointsForKde.append( allPoints )
            


        def getBoxplotDataForPoint(x):
            return dict(med=medians[x], q1=q75[x], q3=q25[x], whislo=q1iqr[x], whishi=q3iqr[x], fliers=())

        if showBoxplots:
            # positions: 1.0 -> -90, 1.5 -> -40, 2.0 -> 10
            bp = ax.bxp( [getBoxplotDataForPoint(10)],
                         positions=(2.0+boxplotXoffset,),
                         widths=boxplotWidth,
                         vert=True,
                         patch_artist=True,
                         medianprops=dict(color='black'),
                         #manage_ticks=False,
                         zorder=2 )
            for patch in bp['boxes']:
                patch.set(facecolor=color)
                #patch.set(linecolor="#ffffff")

        # Add a legend line for this plot
        legendLines.append( Line2D([0], [0], color=color, lw=5.0) )
        legendCaptions.append(label)
            
        
        
        #plt.boxplot( profiles.loc[:,boxplotTicks].transpose(), positions=xvals )
        # bp = plt.boxplot( profiles.loc[:,boxplotTicks].transpose(),
        #                   positions=list(boxplotTicks),
        #                   widths=2,
        #                   flierprops=dict(markersize=2),
        #                   patch_artist=True, ax=ax)
        #plt.axvline(x=1.0,  color="green", zorder=9)
        #plt.axvline(x=25.0, color="blue", zorder=9)


        # for patch in bp['boxes']:
        #    patch.set(facecolor="#2288f0")
        #    patch.set(linecolor="#ffffff")
        #plt.broken_barh( xranges=[(x-0.5,x+0.5) for x in boxplotTicks],
        #                 yrange=(list(q25.values.flat), list(q75.values.flat)) )


    #boxplotColors = dict(boxes='Black', whiskers='Black', medians='Black', caps='Black')
    if category is None:
        if not fileBaseName is None:
            baseName = "dlfe_{}_dists".format(fileBaseName)
        else:
            baseName = "dlfe_dists"

        # bp = profiles.loc[:,list(boxplotTicks)].boxplot( widths=boxplotWidth,
        #                                                  vert=True,
        #                                                  flierprops=dict(markersize=2),
        #                                                  boxprops=dict(color='black'),
        #                                                  medianprops=dict(color='black'),
        #                                                  #manage_ticks=False,
        #                                                  patch_artist=True,
        #                                                  #color=boxplotColors,
        #                                                  zorder=2,
        #                                                  ax=ax)

        
        plotCategory(profiles, plotIQR=False, plotOutliers=False, color=colors[0], label=defaultLabel)

    else:
        baseName = "dlfe_dists_{}".format(category)

        # bp = profiles.loc[:,list(boxplotTicks)+['strand']].boxplot( widths=0.2,
        #                                                             vert=True,
        #                                                             flierprops=dict(markersize=2),
        #                                                             boxprops=dict(color='black'),
        #                                                             medianprops=dict(color='black'),
        #                                                             patch_artist=True,
        #                                                             #color=boxplotColors,
        #                                                             by=category,
        #                                                             zorder=2,
        #                                                             ax=ax)
        
        
        #ax.set_aspect(1.0)
        #p =  sns.boxplot( data=profiles.melt(id_vars=[category]), x='variable', y='value', hue=category, fliersize=2.5, linewidth=0.7, ax=ax )
        categories = frozenset(profiles[category].values)
        for n, cat in enumerate(sorted(categories)):
            profilesForCategory = profiles[profiles[category]==cat].iloc[:,:-1]

            xoffset = 0.0
            if len(categories)>1:
                xoffset = (-0.5 + float(n)/(len(categories)-1))*boxplotWidth*0.5

            # bp = profiles.loc[:,list(boxplotTicks)].boxplot( widths=boxplotWidth/len(categories),
            #                                                  vert=True,
            #                                                  flierprops=dict(markersize=2),
            #                                                  boxprops=dict(color='black'),
            #                                                  medianprops=dict(color='black'),
            #                                                  #manage_ticks=False,
            #                                                  patch_artist=True,
            #                                                  positions=[x+xoffset for x in boxplotTicks],
            #                                                  #color=boxplotColors,
            #                                                  zorder=2,
            #                                                  ax=ax)
            # ax.boxplot( profiles.loc[:,list(boxplotTicks)],
            #             vert=True,
            #             positions=[x+xoffset for x in boxplotTicks],
            #             widths=boxplotWidth/len(categories),
            #             patch_artist=True,
            #             #manage_ticks=False,
            #             flierprops=dict(markersize=2),
            #             medianprops=dict(color='black'),
            #             boxprops=dict(color='black'),
            #             zorder=2 )
            # positions: 1.0 -> -90, 1.5 -> -40, 2.0 -> 10
            
            plotCategory(profilesForCategory, plotIQR=False, plotOutliers=False, color=colors[n], boxplotXoffset=xoffset, boxplotWidth=boxplotWidth/len(categories), label=cat)

    #
    # Plot highlighted species

    if highlightSpecies:
        
        for n, taxid in enumerate(highlightSpecies):
            binomicName = getSpeciesName(taxid).split(' ')
            shortName = "{}. {}".format(binomicName[0][0], binomicName[1])
            ax.plot( xvals2, profiles.loc[taxid,:], label=shortName, c=speciesColors[n], linewidth=2.0, zorder=3 )

            # Add legend lines for this species
            legendLines.append( Line2D([0], [0], color=speciesColors[n], lw=2.0) )
            legendCaptions.append(shortName)
            

    Z  = None
    Zi = None
    if not pointsForKde.empty:
        # DEBUG ONLY ####  DEBUG ONLY ####  DEBUG ONLY ####  DEBUG ONLY ####  DEBUG ONLY ####  DEBUG ONLY #
        #NN = int(pointsForKde.shape[0]*0.8)
        #pointsForKde = pointsForKde.append( pd.DataFrame( {'variable':[-60]*NN, 'value':[-2.0]*NN} ) )
        # DEBUG ONLY ####  DEBUG ONLY ####  DEBUG ONLY ####  DEBUG ONLY ####  DEBUG ONLY ####  DEBUG ONLY #
        #x, y = np.mgrid[ xrange[0]:xrange[1]:1000j, yrange[0]:yrange[1]:500j ]
        yvals = np.linspace( yrange[0], yrange[1], 500 )
        #positions = np.vstack([x.ravel(), y.ravel()]) # coordinates for all "pixels"
        sampleSizeForKde = min( 10000, pointsForKde.shape[0] )

        # 2d KDE
        #kernel = gaussian_kde( pointsForKde.loc[:,['variable','value']].sample( n=sampleSizeForKde ).T.values.astype(float), bw_method=0.06 )
        #Z = np.reshape( kernel.evaluate( positions ).T, x.shape )
        # 1d KDE
        Z = np.zeros( (0,500) )
        
        for x in xvals:
            data = pointsForKde.loc[ pointsForKde['variable'] == x, ['value'] ].T.values.astype(float)
            if not data.any():
                Z = np.vstack( ( Z, np.zeros( (1,500) ) ) )
            else:
                xs = pointsForKde.loc[ pointsForKde['variable'] == x, ['value'] ].T.values.astype(float)
                xs = xs[ np.logical_not( np.isnan( xs ) ) ]  # remove NANs
                kernel = gaussian_kde( xs, bw_method=0.5 ) #0.4 )
                Z = np.vstack( ( Z, np.expand_dims( kernel.evaluate( yvals ), 0 ) ) )
                print("{} {}".format(x, kernel.factor))

        #Z = 1.0 - Z/np.max(Z)  # inverse, normalize, and stretch the values
        Z = 1.0 - Z # /np.max(Z)  # inverse, normalize, and stretch the values
        #Zi = np.dstack( (np.rot90(Z),)*3 )
        Zi = np.rot90(Z)
        ax.imshow( Zi, extent=[transformToFigureXscale(xrange[0]), transformToFigureXscale(xrange[1]), yrange[0], yrange[1]], cmap=plt.cm.Blues_r )



            

    plt.axvline( x=transformToFigureXscale(0.0),   color="black", linewidth=1.0, alpha=1.0, zorder=2 )
    plt.axhline( y=0.0,                            color="black", linewidth=1.0, alpha=1.0, zorder=2)
    ax.set_ylim( yrange )
    ax.set_xlim( (transformToFigureXscale(xrange[0]), transformToFigureXscale(xrange[1])) )
    xticks = range(-80,81,20)
    ax.set_xticks(      [transformToFigureXscale(x)    for x in xticks] )
    ax.set_xticklabels( [toStrWithUnicodeMinusSign(x)  for x in xticks] )
    ax.grid( axis='y' )
    plt.legend(legendLines, legendCaptions)    


    plt.xlabel( "Distance from stop codon (nt)" )
    plt.ylabel( "∆LFE" )
    
    ax.set_aspect(aspect)
    plt.savefig( "{}.pdf".format(baseName), bbox_inches='tight', dpi=100 )
    plt.savefig( "{}.svg".format(baseName), bbox_inches='tight', dpi=100 )

    #ax.set_aspect(1.0)
    #plt.savefig( "{}_asp1.0.pdf".format(baseName), bbox_inches='tight', dpi=100 )
    #ax.set_aspect(2.0)
    #plt.savefig( "{}_asp2.0.pdf".format(baseName), bbox_inches='tight', dpi=100 )
    
    plt.close(fig)


    # if category is None:
    #     fig, ax = plt.subplots()
    #     data= pd.melt(profiles.reset_index(), id_vars=['index'] )
    #     data.plot( x="variable", y="value", color="black", alpha=0.2, ax=ax )
    #     plt.savefig( "{}_dbg.pdf".format(baseName), bbox_inches='tight', dpi=100 )


    if not Z is None:
        fig, ax = plt.subplots()
        ax.imshow( Zi, extent=[transformToFigureXscale(xrange[0]), transformToFigureXscale(xrange[1]), yrange[0], yrange[1]], cmap=plt.cm.Blues_r )
        ax.set_aspect(0.2)
        plt.savefig( "{}_dbg2.pdf".format(baseName), bbox_inches='tight', dpi=100 )
        plt.close(fig)

        fig, ax = plt.subplots()
        for bw in (0.1, 0.3, 0.8, 1.6):
            xs = pointsForKde.loc[ pointsForKde['variable'] == 10, ['value'] ].T.values.astype(float)
            xs = xs[ np.logical_not( np.isnan( xs ) ) ]  # remove NANs
            kernel = gaussian_kde( xs, bw_method=bw )
            Z = kernel.evaluate( yvals )
            ax.plot( yvals, Z, label=str(bw) )
        ax.legend()
        plt.savefig( "{}_dbg3.pdf".format(baseName), bbox_inches='tight', dpi=100 )
        plt.close(fig)
        
            
            

                
        

    print("===="*10)
    print("N = {}".format( profiles.shape[0] ))
    print("===="*10)


def plotPanelizedDLFEComparison( profiles, category=None, xticks=None, xticklabels=None, highlightSpecies=(), boxplotTicks=None, yrange=(-4,1), boxplotWidth=0.05, xrange=(-100,100), colors=('#2040f0',), defaultLabel="All", showBoxplots=True ):
    if category is None:
        baseName="dlfe_panels"
    else:
        baseName="dlfe_panels_{}".format(category)

    g = sns.FacetGrid( data= pd.melt(profiles.reset_index(), id_vars=['index','operon_pos']), row='index', col="operon_pos", hue="operon_pos", sharex=True, sharey=True )
    g.map( plt.plot, "variable", "value" )
        
    
    plt.savefig( "{}.pdf".format(baseName), bbox_inches='tight', dpi=100 )
    plt.savefig( "{}.svg".format(baseName), bbox_inches='tight', dpi=100 )

    
def getHeatmaplotProfilesValuesRange(data, dummy1=None, dummy2=None):

    #keysInOrder = data.keys()[:]
    #if not order is None:
    #    keysInOrder.sort( key=lambda x:order(x) )
    keysInOrder = list(data.keys())
    

    # Find the overall range that will be used to normalize all values
    # Todo - allow some "over-exposure" to expand the mid range at the cost of losing detail at the extremes
    valuesRange = [0.0, 0.0]
    for taxId in keysInOrder:
        series = data[taxId]

        currMin = min(series)
        currMax = max(series)

        if(currMin < valuesRange[0] ):
            valuesRange[0] = currMin
        if(currMax > valuesRange[1] ):
            valuesRange[1] = currMax
    print(valuesRange)
    #assert(valuesRange[0] < 0.0)
    #assert(valuesRange[1] > 0.0)
    maxRange = max(-valuesRange[0], valuesRange[1])
    
    return (-maxRange, maxRange)  # return the normalized range used

    # fig, axes = plt.subplots(nrows=len(data), ncols=2, sharex='col') #, gridspec_kw={'width_ratios': [4,1]})
    # plt.grid(False)
    # cmapNormalizer = CenterPreservingNormlizer(-maxRange, maxRange)
    # for ax, taxId in zip([x[0] for x in axes], keysInOrder):
    #     series = data[taxId]

    #     imdata = np.array(series)
    #     imdata = np.vstack((imdata,imdata))  # pretty crude trick borrowed from matplotlib examples...

    #     #ax.axis(xmin=series.index[0], xmax=series.index[-1])

    #     ax.imshow( imdata, cmap='coolwarm', aspect='auto', norm=cmapNormalizer )
            
    #     pos = list(ax.get_position().bounds)

    #     taxname = getSpeciesFileName(taxId)
    #     taxDescriptor = "%s.%s" % (taxname[0], taxname[1:9])
    #     fig.text(pos[0]-0.01, pos[1]+pos[3]/2., taxDescriptor, va='center', ha='right', fontsize=8)
        
    #     #ax.set_title(taxId)
    #     ax.set_yticks(())
    #     #ax.tick_params

    # #plt.colorbar(im, ax=axes[0], norm=cmapNormalizer, orientation='horizontal')
    # #plt.colorbar(im, cax=axes[-1], orientation='horizontal')
    # cbarRange = np.expand_dims( np.linspace( valuesRange[0]+0.001, valuesRange[1], 200 ), 2)

    # cbx = plt.subplot2grid((len(data),2), (0,1), rowspan=len(data))
    # #cbx.imshow( np.hstack((cbarRange, cbarRange)), aspect='auto', cmap='coolwarm', norm=cmapNormalizer)
    # #cbx.set_xticks(())


    # corrDataForPlotting = corrData.rename( columns={'spearman_smfe_gc_rho':'GC%', 'spearman_smfe_Nc_rho':'ENc', 'spearman_smfe_CAI_rho':'CAI', 'spearman_smfe_Fop_rho':'Fop' } )
    # del corrDataForPlotting['spearman_smfe_gc_pval']
    # del corrDataForPlotting['spearman_smfe_Nc_pval']
    # del corrDataForPlotting['spearman_smfe_CAI_pval']
    # del corrDataForPlotting['spearman_smfe_Fop_pval']
    # orderdf = pd.DataFrame({'order':range(len(keysInOrder))}, index=keysInOrder)
    # df2 = pd.merge(corrDataForPlotting, orderdf, left_index=True, right_index=True, how='inner')
    # df2.sort_values(by=['order'])
    # del df2['order']

    # corrsHeatmap = sns.heatmap(df2, annot=True, fmt=".2g", ax=cbx)
    # #corrsHeatmap.savefig("heatmap_profile_correlations.pdf")
    # #corrsHeatmap.savefig("heatmap_profile_correlations.svg")

    # plt.savefig("heatmap_profile_test.pdf", orientation='portrait')
    # plt.savefig("heatmap_profile_test.svg", orientation='portrait')
    # plt.close(fig)

    # return (-maxRange, maxRange)  # return the normalized range used


def plotCorrelations(data, _labels, group_func=None, order=None):
    #fig, axes = plt.subplots(nrows=len(data), ncols=2, sharex='col') #, gridspec_kw={'width_ratios': [4,1]})

    keysInOrder = list(data.keys())[:]
    keysInOrder.sort( key=lambda x:order(x) )

    #map  = sns.heatmap()
    #keysInOrder = data.keys()

def scatterPlotWithColor(taxId, profileId, shuffleType, data, xvar, yvar, colorvar, title):
    fig, ax1 = plt.subplots()

    #data.plot(x=xvar, y=yvar, c=colorvar, size=3, ax=ax1, kind='scatter')
    data[(data[colorvar]>=0.25) & (data[colorvar]<=0.75)].plot(x=xvar, y=yvar, c='white', s=3, ax=ax1, kind='scatter')
    data[data[colorvar].isnull()].plot(x=xvar, y=yvar, c='white', s=3, ax=ax1, kind='scatter')
    data[data[colorvar]>0.75].plot(x=xvar, y=yvar, c='red', s=3, ax=ax1, kind='scatter')
    data[data[colorvar]<0.25].plot(x=xvar, y=yvar, c='blue', s=3, ax=ax1, kind='scatter')

    top20 = data.sort_values(by=colorvar, ascending=False).iloc[:20]
    for i in range(20):
        plt.annotate(s=top20.iloc[i]['protid'], xy=( top20.iloc[i][xvar], top20.iloc[i][yvar]), fontsize=2 )

    top20.plot(x=xvar, y=yvar, c='orange', s=10, alpha=0.7, ax=ax1, kind='scatter')

    
    bottom20 = data.sort_values(by=colorvar, ascending=False).iloc[-20:]
    for i in range(20):
        plt.annotate(s=bottom20.iloc[i]['protid'], xy=( bottom20.iloc[i][xvar], bottom20.iloc[i][yvar]), fontsize=2 )

    bottom20.plot(x=xvar, y=yvar, c=(0.2,0.7,1.0), s=10, alpha=0.7, ax=ax1, kind='scatter')
    

    plt.annotate(s="n= %d"  % len(data),                                                 xy=(0.3, 4),  fontsize=6 )
        

    #plt.xlim([.25,.75])
    plt.xlabel(xvar.replace('_',' '))
    plt.ylabel(yvar.replace('_',' '))
    plt.title( title % getSpeciesName(taxId) )
    plt.grid(True)
    #plt.legend(loc=(0,1), scatterpoints=1, ncol=3, fontsize='small')

    plt.savefig("mfe_v2_40nt_genelevel_%s_vs_%s_with_%s_%s_t%d_%s.pdf" % (yvar, xvar, colorvar, profileId, shuffleType, getSpeciesFileName(taxId)))
    plt.savefig("mfe_v2_40nt_genelevel_%s_vs_%s_with_%s_%s_t%d_%s.svg" % (yvar, xvar, colorvar, profileId, shuffleType, getSpeciesFileName(taxId)))
    plt.close(fig)
    



def plotMFEProfileByPA(taxId, profileId, data):
    fig, ax1 = plt.subplots()

    data[['native', 'shuffled']].plot(ax=ax1)
    data[['native_pa_low', 'native_pa_med', 'native_pa_high']].plot(ax=ax1)
    data[['shuffled75', 'shuffled25']].plot(ax=ax1, style='--')

    speciesName = getSpeciesName(taxId)

    plt.title(speciesName)

    plt.xlabel('Position (nt, window start, from cds start)')

    ax1.set_title("Mean LFE for %s" % speciesName)
    ax1.set_ylabel('Mean LFE')
    ax1.legend(fontsize=8)
    ax1.grid(True)


    #profileId = "tbd" # str(args.profile.ProfileId).replace(':', '-')
    plt.savefig("mfe_v2_40nt_cds_%s_%s_by_pa.pdf" % (profileId, getSpeciesFileName(taxId)) )
    plt.savefig("mfe_v2_40nt_cds_%s_%s_by_pa.svg" % (profileId, getSpeciesFileName(taxId)) )
    plt.close(fig)


def scatterPlotWithKernel(taxId, profileId, data, xvar, yvar, title):
    g = sns.jointplot( xvar, yvar, data=data, kind="kde", dropna=True, ylim=(-5 ,5), xlim=(0.3, 0.7), gridsize=100 )
    g.savefig("mfe_v2_40nt_genelevel_%s_vs_%s_%s_%s.pdf" % (yvar, xvar, profileId, getSpeciesFileName(taxId)))
    g.savefig("mfe_v2_40nt_genelevel_%s_vs_%s_%s_%s.svg" % (yvar, xvar, profileId, getSpeciesFileName(taxId)))


def scatterPlotWithKernel2(taxId, profileId, data, xvar, yvar, title):
    data = data.copy()
    data.dropna(subset=(xvar, yvar), inplace=True)

    fig, ax1 = plt.subplots()
    #data.plot(x=xvar, y=yvar, ax=ax1, kind='scatter')
    ax = sns.kdeplot( data[xvar], data[yvar], cmap="Blues", shade=True, shade_lowest=True, legend=True)
    blue = sns.color_palette("Blues")[-2]



    ################################

    xvals = data[xvar]
    yvals = data[yvar]
    #print(data.head())

    # Linear correlation and factors
    pearson = pearsonr(xvals, yvals)
    spearman = spearmanr(xvals, yvals)
    kendall = kendalltau(xvals, yvals)
    l = linregress(xvals, yvals)

    min_xvals = 0.4
    max_xvals = 0.7
    abline_x = np.arange(min_xvals, max_xvals*1.01, (max_xvals-min_xvals) )

    abline_y = abline_x * l.slope + l.intercept
    plt.plot(abline_x, abline_y, '--')

    #topr = max(yvals)*1.05
    topr = 100
    #left = min(xvals)
    left = min_xvals
    scaler = topr / 20
    # plot the linear approximation
    plt.annotate(s="Pearson r: %1.3f (p<%g)"  % (pearson[0], pearson[1]),                 xy=(left, topr-scaler*1),  fontsize=6 )
    plt.annotate(s="Pearson $r^2$: %1.3f"  % (pearson[0]**2,),                              xy=(left, topr-scaler*2),  fontsize=6 )
    plt.annotate(s="Spearman r: %1.3f (p<%g)" % (spearman.correlation, spearman.pvalue),  xy=(left, topr-scaler*3),  fontsize=6 )
    plt.annotate(s="Kendall's tau: %1.3f (p<%g)" % (kendall.correlation, kendall.pvalue), xy=(left, topr-scaler*4),  fontsize=6 )
    plt.annotate(s="n= %d"  % len(yvals),                                                 xy=(left, topr-scaler*5),  fontsize=6 )


    plt.xlim([0.5,0.7])
    plt.ylim([-10,10])
    



    ################################3


    

    #plt.annotate(s="n= %d"  % len(data),                                                 xy=(0.3, 4),  fontsize=6 )
        

    #plt.xlim([.25,.75])
    plt.xlabel(xvar.replace('_',' '))
    plt.ylabel(yvar.replace('_',' '))
    plt.title( title % getSpeciesName(taxId) )
    plt.grid(True)
    #plt.legend(loc=(0,1), scatterpoints=1, ncol=3, fontsize='small')

    plt.savefig("mfe_v2_40nt_genelevel_%s_vs_%s_%s_%s.pdf" % (yvar, xvar, profileId, getSpeciesFileName(taxId)))
    plt.savefig("mfe_v2_40nt_genelevel_%s_vs_%s_%s_%s.svg" % (yvar, xvar, profileId, getSpeciesFileName(taxId)))
    plt.close(fig)


short_names = set()
def shortenTaxName(name):
    currLength=4
    
    if name.startswith("Candidatus "): # drop 'Candidatus' prefix
        name = name[11:]
        
    while(currLength <= len(name)):
        candidate = name[:currLength]
        if not candidate in short_names:
            short_names.add(candidate)
            return candidate
        currLength += 1

    #raise Exception("Failed to shorten name '%s'" % name)
    
    # Try adding numerical indices to resolve ambiguities
    idx=1
    while(True):
        candidate = "%s%d" % (name[:4], idx)
        if not candidate in short_names:
            short_names.add(candidate)
            return candidate
        idx += 1

def calcWilcoxonPvalue_method1(df):
    difs = np.array(df.native - df.shuffled)
    direction = np.sign(np.mean(difs))

    pval = wilcoxon(difs).pvalue
    
    return log10(pval) * direction * -1

def calcWilcoxonPvalue_method2(df2):
    assert(df2.ndim==2)

    df2 = df2[~df2['delta'].isnull()]

    direction = np.sign(np.mean(df2['delta']))
    pval = wilcoxon(df2['delta']).pvalue

    if( pval>0.0 ):
        return log10(pval) * direction * -1
    elif( pval==0.0):    # I think exact comparison to 0.0 is safe with floating point numbers
        return -320.0      * direction * -1
    else:
        assert(False)




def getShortTaxName(taxId):
    return getSpeciesFileName(taxId)

def loadProfileData(files):
    xdata = []
    ydata = []
    ydata_nativeonly = []
    ydata_shuffledonly = []
    labels = []
    groups = []
    filesUsed = 0
    biasProfiles = {}
    wilcoxonDLFEZeroData = {}
    wilcoxonDLFE50_100Data = {}
    sequenceNativeProfilesData = {}
    sequenceRandomizedProfilesData = {}

    dfProfileCorrs = pd.DataFrame( { "spearman_smfe_gc_rho":   pd.Series(dtype='float'),
                                     "spearman_smfe_gc_pval":  pd.Series(dtype='float'),
                                     "spearman_smfe_Nc_rho":   pd.Series(dtype='float'),
                                     "spearman_smfe_Nc_pval":  pd.Series(dtype='float'),
                                     "spearman_smfe_CAI_rho":  pd.Series(dtype='float'),
                                     "spearman_smfe_CAI_pval": pd.Series(dtype='float'),
                                     "spearman_smfe_Fop_rho":  pd.Series(dtype='float'),
                                     "spearman_smfe_Fop_pval": pd.Series(dtype='float') } )

    summaryStatistics = pd.DataFrame({
        'tax_name':pd.Series(dtype='str'),
        'short_tax_name':pd.Series(dtype='str'),
        'tax_id':pd.Series(dtype='int'),
#        'genomic_gc':pd.Series(dtype='float'),
        'tax_group':pd.Series(dtype='str'), # TODO: change to categorical data; Categorical([], categories=('Bacteria', 'Archaea', 'Fungi', 'Plants'), ordered=False)
        'CDSs_included':pd.Series(dtype='int'),
        'profileElements':pd.Series(dtype='int')
#        'optimal_temperature':pd.Series(dtype='float'),
#        'temperature_range':pd.Categorical([]),
#        'mean_delta_lfe':pd.Series(dtype='float'),
#        'paired_fraction':pd.Series(dtype='float'),
#        'gene_density':pd.Series(dtype='float')
    })

    rl = RateLimit(10)

    for h5 in files:
        with pd.io.pytables.HDFStore(h5, 'r') as store:
            for key in list(store.keys()):
                if key[:4] != "/df_":
                    continue

                dfHeader = key.split('_')
                taxId = int(dfHeader[1])
                taxName = getShortTaxName(taxId)
                #taxGroup = data_helpers.getSpeciesTaxonomicGroup(taxId)
                taxGroup = getTaxonomicGroupForSpecies(taxId)
                longTaxName = getSpeciesName(taxId)
                shortTaxName = shortenTaxName(taxName)

                df = store[key]
                df = df.iloc[:-1]  # remove the last value (which is missing)

                deltas_df = None
                if "/deltas_"+key[4:] in store:
                    deltas_df = store["/deltas_"+key[4:]]

                genes_df = None
                if "/deltas_"+key[4:] in store:
                    genes_df = store["/deltas_"+key[4:]]

                summary_df = None
                if "/statistics_"+key[4:] in store:
                    summary_df = store["/statistics_"+key[4:]]

                profileCorrelations_df = None
                if "/profiles_spearman_rho_"+key[4:] in store:
                    profileCorrelations_df = store["/profiles_spearman_rho_"+key[4:]]

                wilcoxonDlfeZero = None
                if '/wilcoxon_dlfe_'+key[4:] in store:
                    wilcoxonDlfeZero = store['/wilcoxon_dlfe_'+key[4:]]
                    wilcoxonDLFEZeroData[taxId] = wilcoxonDlfeZero

                wilcoxonDlfe50_100 = None
                if '/wilcoxon_dlfe50_100_'+key[4:] in store:
                    wilcoxonDlfe50_100 = store['/wilcoxon_dlfe50_100_'+key[4:]]
                    wilcoxonDLFE50_100Data[taxId] = wilcoxonDlfe50_100

                sequenceNativeProfiles = None
                if '/sequence_native_lfe_'+key[4:] in store:
                    sequenceNativeProfiles = store['/sequence_native_lfe_'+key[4:]]
                    sequenceNativeProfilesData[taxId] = sequenceNativeProfiles
                    
                sequenceRandomizedProfiles = None
                if '/sequence_randomized_lfe_'+key[4:] in store:
                    sequenceRandomizedProfiles = store['/sequence_randomized_lfe_'+key[4:]]
                    sequenceRandomizedProfilesData[taxId] = sequenceRandomizedProfiles

                    
                df['MFEbias'] = pd.Series(df['native']-df['shuffled'], index=df.index)
                dfMFEbias = df['MFEbias']

                biasProfiles[taxId] = dfMFEbias

                meanDeltaLFE = np.mean(dfMFEbias)

                cdsCount = int(summary_df.iloc[0]['cds_count'])
                print(taxId)
                #assert(cdsCount >= 100)
                if cdsCount<100:
                    print("Warning: only {} results for species {}".format(cdsCount, taxId))
                #print("--------")
                # firstPos = (deltas_df['pos'].min())
                # lastPos = (deltas_df['pos'].min())
                # numSamplesIncludedInProfile = min( pd.isnull(deltas_df[deltas_df['pos']==firstPos]['delta']).sum(),
                #                                    pd.isnull(deltas_df[deltas_df['pos']==lastPos ]['delta']).sum() )

                # #print( "{:.2}".format(float(numSamplesIncludedInProfile) / cdsCount ))
                # if float(numSamplesIncludedInProfile) / cdsCount < 0.5:
                #     print( "{:.2}".format(float(numSamplesIncludedInProfile) / cdsCount ))
                #     print("Warning: {} (taxId={}): not enough data is available".format(longTaxName, taxId))
                #     #continue  # Skip sequences with very limited data available
                
                #meanGC = species_selection_data.findByTaxid(taxId).iloc[0]['GC% (genome)']
                #meanGC = getGenomicGCContent(taxId)  # this is actually the genomic GC% (not CDS only)

                # Fetch temperature data for this species (if available)
                # optimalTemperatureData = getSpeciesProperty( taxId, 'optimum-temperature')
                # optimalTemperature = None
                # if not optimalTemperatureData[0] is None:
                #     optimalTemperature = float(optimalTemperatureData[0])

                # temperatureRangeData = getSpeciesProperty( taxId, 'temperature-range')
                # temperatureRange = None
                # if not temperatureRangeData[0] is None:
                #     temperatureRange = temperatureRangeData[0]
                # else:
                #     temperatureRange = "Unknown"

                # pairedFractionData = getSpeciesProperty( taxId, 'paired-mRNA-fraction')
                # pairedFraction = None
                # if not pairedFractionData[0] is None:
                #     pairedFraction = float(pairedFractionData[0])

                    
                # genomeSizeData = getSpeciesProperty( taxId, 'genome-size-mb')
                # genomeSize = None
                # if not genomeSizeData[0] is None:
                #     genomeSize = float(genomeSizeData[0])

                # proteinCountData = getSpeciesProperty( taxId, 'protein-count')
                # proteinCount = None
                # if not proteinCountData[0] is None:
                #     proteinCount = int(proteinCountData[0])

                # geneDensity = None
                # if( (not genomeSize is None) and (not proteinCount is None)  ):
                #     geneDensity = float(proteinCount)/genomeSize

                    
                summaryStatistics = summaryStatistics.append(pd.DataFrame({
                    'tax_name':pd.Series([taxName]),
                    'short_tax_name':pd.Series([shortTaxName]),
                    'long_tax_name':pd.Series([longTaxName]),
                    'tax_id':pd.Series([taxId], dtype='int'),
#                    'genomic_gc':pd.Series([meanGC]),
                    'tax_group':pd.Series([taxGroup]),
                    'CDSs_included':pd.Series([cdsCount], dtype='int')
#                    'optimal_temperature':pd.Series([optimalTemperature], dtype='float'),
#                    'temperature_range':pd.Categorical([temperatureRange]),
#                    'mean_delta_lfe':pd.Series([meanDeltaLFE], dtype='float'),
#                    'paired_fraction':pd.Series([pairedFraction], dtype='float'),
#                    'gene_density':pd.Series([geneDensity], dtype='float')
                }))

                dfProfileCorrs = dfProfileCorrs.append( profileCorrelations_df )

                # Format:

                #         gc  native  position  shuffled
                # 1    0.451  -4.944         1    -5.886
                # 2    0.459  -5.137         2    -6.069
                # 3    0.473  -5.349         3    -6.262
                filesUsed += 1

                #print(df.shape)

                meanGC = np.mean(df.gc)
                xdata.append(meanGC)

                #meanE = np.mean(df.native - df.shuffled)
                #ydata.append(meanE)

                if not deltas_df is None:
                    dirpval = calcWilcoxonPvalue_method2(deltas_df)
                    ydata.append(dirpval)

                #print(df.head())

                #print(df.native)
                #print(df.shuffled)

                meanE_nativeonly = np.mean(df.native)
                ydata_nativeonly.append(meanE_nativeonly)

                meanE_shuffledonly = np.mean(df.shuffled)
                ydata_shuffledonly.append(meanE_shuffledonly)

                labels.append( taxName )

                #groups.append( choice(('Bacteria', 'Archaea', 'Fungi', 'Plants')) )   # Testing only!!!
                groups.append( taxGroup )

                if( rl() ):
                    print("Loaded %d profiles (%.2g%%)" % (filesUsed, float(filesUsed)/len(files)*100))

    return (xdata, ydata, ydata_nativeonly, ydata_shuffledonly, labels, groups, filesUsed, biasProfiles, dfProfileCorrs, summaryStatistics, wilcoxonDLFEZeroData, wilcoxonDLFE50_100Data, sequenceNativeProfilesData, sequenceRandomizedProfilesData)

def loadPhylosignalProfiles( phylosignalFile ):
    df = None
    with open(phylosignalFile, 'r') as csvfile:
        df = pd.read_csv(csvfile, sep=',', dtype={"":np.int}, index_col=0 )
    return df


def getTaxName(taxId):
    return getSpeciesFileName(taxId)


_shortTaxNames = None
def getSpeciesShortestUniqueNamesMapping_memoized():
    global _shortTaxNames
    if( _shortTaxNames is None ):
        _shortTaxNames = getSpeciesShortestUniqueNamesMapping()
    return _shortTaxNames



def addPanel(ctx, filename, h, w):
    ctx.save()

    im1 = cairo.ImageSurface.create_from_png( file( filename, 'r') )

    # Set origin for this layer
    ctx.translate( 0, 0 )

    imgpat = cairo.SurfacePattern( im1 )

    imh = im1.get_height()
    imw = im1.get_width()

    #scale_w = imw/w
    #scale_h = imh/h
    #compromise_scale = max(scale_w, scale_h)
    

    # Scale source image
    #scaler = cairo.Matrix()
    #scaler.scale(compromise_scale, compromise_scale)
    #imgpat.set_matrix(scaler)
    #imgpat.set_filter(cairo.FILTER_BEST)


    ctx.set_source(imgpat)

    ctx.rectangle( 0, 0, w, h )

    ctx.fill()
    ctx.restore()

def overlayImages(images, outputFile):
    fo = file(outputFile, 'w')

    h=2400
    w=1800

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    ctx = cairo.Context( surface )

    # ------------------------- Background -------------------------
    ctx.set_source_rgb( 1.0, 1.0, 1.0 )
    ctx.paint()

    for image in images:
        addPanel(ctx, image, h, w)

    surface.write_to_png(fo)

    surface.finish()


def estimateModeUsingKDE(xs):
    xs = np.expand_dims(xs, 1)
    assert(xs.ndim==2)

    bandwidths = 10 ** np.linspace(-2.0, 1, 500)
    
    cv = KFold(len(xs), n_folds=10)
    #cv = LeaveOneOut(len(x1))
    
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=cv )
    grid.fit(xs)
    bw = grid.best_params_['bandwidth']
    
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(xs)  # calculate the KDE
    print(xs.shape)

    x_d = np.linspace( min(xs), max(xs), 500 )  # This must cover the range of values.. TODO - fix this...
    print( np.expand_dims(x_d, 1).shape)
    
    logprob = kde.score_samples( np.expand_dims(x_d, 1) )  # score the KDE over a range of values

    pos = np.argmax(logprob)
    peak = x_d[pos]
    peakVal = logprob[pos]
    assert(all(logprob <= peakVal))
    return (peak, peakVal)
    

class LayerConfig(object):
    _defaults = dict(showAxes=False, showDensity=False, showDists=False, showProfiles=False, showHighlights=False, showComponents=False, showLoadingVectors=False, showTickMarks=False, debug=False )

    def __init__(self, **kw):
        self._kw = LayerConfig._defaults.copy()
        
        self._kw.update(**kw)

    def __str__(self):
        return str(self._kw)

    def __getattr__(self, name):
        return self._kw.get(name)

# Source: https://stackoverflow.com/a/13849249
def unitVector(v):
    return v / np.linalg.norm(v)

# Source: https://stackoverflow.com/a/13849249
def angleBetween(v1,v2):
    v1_u = unitVector(v1)
    v2_u = unitVector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

assert(abs(angleBetween(np.array([1,0,0,0]),np.array([-1,0,0,0])) - np.pi  ) < 1e-6)
assert(abs(angleBetween(np.array([1,0,0,0]),np.array([ 0,1,0,0])) - np.pi/2) < 1e-6)
assert(abs(angleBetween(np.array([1,0,0,0]),np.array([10,0,0,0])) - 0.0    ) < 1e-6)

def testPCArobustness(vectors, repeat=1000, sampleSize=500):

    def testIteration():
        ixs = np.random.randint(vectors.shape[0], size=sampleSize)
        sample = vectors.take(ixs, axis=0)
        
        # Perform the PCA
        pca = decomposition.PCA()
        pca.n_components = 3  # force 3 components
        pca.fit(sample)


        v1 = pca.components_[0,:]
        v2 = pca.components_[1,:]
        v3 = pca.components_[2,:]

        assert( abs(angleBetween(v1,v2) - np.pi/2) <= 1e-6 )
        assert( abs(angleBetween(v1,v3) - np.pi/2) <= 1e-6 )
        assert( abs(angleBetween(v2,v3) - np.pi/2) <= 1e-6 )

        c_0nt_2d   = np.array( (pca.components_[0, 0], pca.components_[1, 0]) )
        c_250nt_2d = np.array( (pca.components_[0,25], pca.components_[1,25]) )
        c_50nt_2d  = np.array( (pca.components_[0, 5], pca.components_[1, 5]) )

        return (angleBetween(c_0nt_2d, c_250nt_2d),
                angleBetween(c_0nt_2d, c_50nt_2d),
                pca.explained_variance_ratio_[0],
                pca.explained_variance_ratio_[1])

    angles1_2 = []
    angles1_3 = []
    varexp1 = []
    varexp2 = []
    for n in range(repeat):
        print(n)
        ret = testIteration()
        print(ret)
        angles1_2.append(ret[0])
        angles1_3.append(ret[1])
        varexp1.append(ret[2])
        varexp2.append(ret[3])

    fig, ax1 = plt.subplots()
    sns.distplot(angles1_2)
    plt.savefig("pca_profiles_robustness_angle_1_2.pdf")
    plt.close(fig)
    
    fig, ax1 = plt.subplots()
    sns.distplot(angles1_3)
    plt.savefig("pca_profiles_robustness_angle_1_3.pdf")
    plt.close(fig)

    fig, ax1 = plt.subplots()
    sns.distplot(varexp1)
    plt.savefig("pca_profiles_robustness_varexp_1.pdf")
    plt.close(fig)

    fig, ax1 = plt.subplots()
    sns.distplot(varexp2)
    plt.savefig("pca_profiles_robustness_varexp_2.pdf")
    plt.close(fig)
    
        

def saveHistogram(data, filename, bins=None):
    fig, ax1 = plt.subplots()
    #sns.distplot(data, ax=ax1, bins=bins, histtype='step', density=False)
    ax1.hist( data, bins=bins )
    plt.title("{} (N={}, median={})".format(filename, data.shape[0], np.median(data)) )
    plt.savefig(filename)
    plt.close(fig)
    
    
    
def PCAForProfiles(biasProfiles, profileValuesRange, profilesYOffsetWorkaround=0.0, profileScale=1.0, fontSize=7, overlapAction="ignore", showDensity=True, highlightSpecies=None, addLoadingVectors=[], debug=False, loadingVectorsScale=5.4, zoom=1.0, legendXpos=0.0, traitValues={}, symbolScale=8.0):
    filteredProfiles = {}
    for key, profile in list(biasProfiles.items()):
        if (not np.any(np.isnan(profile))) and (key in traitValues):
            filteredProfiles[key] = profile
    biasProfiles = filteredProfiles
    
    X = np.vstack(list(biasProfiles.values())) # convert dict of vectors to matrix
    #X = X[~np.any(np.isnan(X), axis=1)]  # remove points containig NaN


    testPCArobustness(X) # create diagnostic plots for the robustness of the PCA solution (that is not generally robust to outliers)
    
    print("Creating PCA plot...")
    

    shortNames = getSpeciesShortestUniqueNamesMapping_memoized()

    # Perform the PCA
    pca = decomposition.PCA()
    pca.fit(X)

    pca.n_components = 3  # force 3 components
    X_reduced = pca.fit_transform(X)
    print(X_reduced.shape)

    debugSymbols = [[0, 1, -1, 0,  0], [0, 0,  0, 1, -1]]   # coordinates for debug symbols ('x')
    
    # Assign dimensions to plot axes
    # TODO - test the other values...
    D0 = 0 # D0 - component to show on Y scale 
    D1 = 1 # D1 - component to show on X scale
    assert(D0!=D1)

    D0_peak = estimateModeUsingKDE( X_reduced[:,D0] )
    D1_peak = estimateModeUsingKDE( X_reduced[:,D1] )
    distPlotsScales = (exp(D1_peak[1])*1.12, exp(D0_peak[1])*1.12)
    print("Peaks: {} {}".format(exp(D1_peak[1]), exp(D0_peak[1])))
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    
    

    def plotPCALayer(layerConfig):
        #fig = matplotlib.figure.Figure(suppressComposite=True)
        #ax = fig.subplots()
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", gridspec_kw={'height_ratios': [1, 8], 'width_ratios': [1, 8]  }, figsize=plt.figaspect(1.333), linewidth=0 )
        for ax in fig.axes: ax.set_autoscale_on(False)
        #ax = ax4

        cmapNormalizer = CenterPreservingNormlizer(profileValuesRange[0], profileValuesRange[1])


        #plt.scatter(X_reduced[:,1], X_reduced[:,0], label=[shortNames[x] for x in biasProfiles.keys()])


        # Calculate general scaling constants
        scaleX = (max(X_reduced[:,D1]) - min(X_reduced[:,D1])) * zoom
        scaleY = (max(X_reduced[:,D0]) - min(X_reduced[:,D0])) * zoom
        ori1 =   max(X_reduced[:,D0])
        ori2 =   max(X_reduced[:,D1])
        #assert(scaleY>scaleX)  # We map the first PCA dimension to the Y axis (since portrait format better fits the figure layout)
        #scaleX = scaleY
        scaleXY = max(scaleX,scaleY)
        imw = scaleXY*0.100*profileScale
        imh = scaleXY*0.012*profileScale
        imofy = profilesYOffsetWorkaround # 0.0 # 0.45 # TODO - FIX THIS

        # Legend position
        #tlx = min(X_reduced[:,D1]) - imw*0.5
        tlx = max(X_reduced[:,D1]) - (imw*legendXpos)
        tly = min(X_reduced[:,D0])
        bry = max(X_reduced[:,D0])

        
        spIndex = Index(bbox=(min(X_reduced[:,D1]), min(X_reduced[:,D0]), max(X_reduced[:,D1]), max(X_reduced[:,D0]) ))

        #zorders = [x[0] for x in sorted(list(enumerate([sqrt((X_reduced[i,D1]-ori2)**2 + (X_reduced[i,D0]-ori1)**2) for i in range(len(X_reduced))])), key=lambda x:x[D1])]

        dinfo = {}

        highlightPatches = []

        if layerConfig.showProfiles:
            # Plot each profile 
            for i, taxId in enumerate(biasProfiles.keys()):
                # Determine the location for this profile
                x = X_reduced[i,D1]
                y = X_reduced[i,D0]

                showProfile = True
                label = shortNames[taxId]

                # is there overlap?
                if overlapAction=="hide":
                    bbox = (x-imw, y-imh, x+imw, y+imh)
                    #print("---"*10)
                    #print("new: {} {}".format(label, bbox))
                    matches = spIndex.intersect( bbox )
                    #for m in matches:
                    #    print("-X- {} {}".format(m, dinfo[m]))

                    if matches and taxId not in highlightSpecies:
                        #print( "Hiding profile at {}".format( (x,y) ) )
                        showProfile = False
                    else:
                        spIndex.insert( label, bbox )
                        dinfo[label] = bbox

                # plot the profile
                #zorder = zorders[i]
                if showProfile:      # skip this if we decided to hide this profile

                    # Show the text label
                    if fontSize > 0:
                        ax4.annotate(label, (x - scaleX*0.03, y + scaleY*0.012), fontsize=fontSize, zorder=100)

                    if taxId in highlightSpecies:
                        rect = Rectangle((bbox[0]-0.03, bbox[1]+0.03+imh), imw*2+0.06, imh*2+0.06, edgecolor="blue", linewidth=4.0, facecolor="none", fill=False, linestyle="solid")
                        highlightPatches.append(rect)


                    # Show the profile tile
                    if True:
                        ax4.imshow( np.array( biasProfiles[taxId] ).reshape(1,-1), cmap='bwr', norm=cmapNormalizer, extent=(x-imw, x+imw, -y-imh+imofy, -y+imh+imofy ), interpolation='bilinear', zorder=2000 )


        # Paint the highlights
        if layerConfig.showHighlights:
            ax4.add_collection(PatchCollection(highlightPatches))

        if layerConfig.showComponents:
            cmapNormalizerForPCAvars = CenterPreservingNormlizer(-1, 1)
            ax4.imshow( pca.components_[0,:].reshape(1,-1), extent=(tlx-2*imw, tlx+0,  tly-imh*2, tly-imh*4 ), norm=cmapNormalizerForPCAvars, cmap='coolwarm', interpolation='bilinear', zorder=300 )
            ax4.imshow( pca.components_[1,:].reshape(1,-1), extent=(tlx-2*imw, tlx+0,  tly      , tly-imh*2 ), norm=cmapNormalizerForPCAvars, cmap='coolwarm', interpolation='bilinear', zorder=300 )
            ax4.imshow( pca.components_[2,:].reshape(1,-1), extent=(tlx-2*imw, tlx+0,  tly+imh*2, tly-imh*0 ), norm=cmapNormalizerForPCAvars, cmap='coolwarm', interpolation='bilinear', zorder=300 )
            ax4.annotate( "V1",                                                (tlx - imw*2.5, bry+imh*2),       fontsize=fontSize*0.9 )
            ax4.annotate( "V2",                                                (tlx - imw*2.5, bry+imh*0),       fontsize=fontSize*0.9 )
            ax4.annotate( "V3",                                                (tlx - imw*2.5, bry-imh*2),       fontsize=fontSize*0.9 )
            ax4.annotate( "V1+V2",                                             (tlx - imw*2.5, bry-imh*4),       fontsize=fontSize*0.9 )
            ax4.annotate( "{:.3g}".format( pca.explained_variance_ratio_[0] ), (tlx - imw*1.0, bry+imh*2), fontsize=fontSize*0.9 )
            ax4.annotate( "{:.3g}".format( pca.explained_variance_ratio_[1] ), (tlx - imw*1.0, bry+imh*0), fontsize=fontSize*0.9 )
            ax4.annotate( "{:.3g}".format( pca.explained_variance_ratio_[2] ), (tlx - imw*1.0, bry-imh*2), fontsize=fontSize*0.9 )
            ax4.annotate( "{:.3g}".format( pca.explained_variance_ratio_[0]+
                                           pca.explained_variance_ratio_[1] ), (tlx - imw*1.0, bry-imh*4), fontsize=fontSize*0.9 )

        if layerConfig.showLoadingVectors:
            isEndReferenced = any([x<0 for x in addLoadingVectors])  # if any of the indices is negative, we tread them referenced to the end of the profile

            for i, c in enumerate(addLoadingVectors):
                print("c = {}".format(c))
                if isEndReferenced:
                    assert(c <= 0)
                    cIdx = X.shape[1] + c - 1
                    print("--> {} (N={})".format(cIdx, X.shape))
                else:
                    cIdx = c

                #ax4.annotate( u"\u0394LFE[{}nt]".format(abs(c)*10),   xy=(0,0), xytext=(pca.components_[D1,cIdx]*loadingScale, pca.components_[D0,cIdx]*loadingScale), fontsize=fontSize, arrowprops=dict(arrowstyle='<-', alpha=0.6, linewidth=2.0, color='red'), color='red', zorder=200 )

                ax4.annotate( "",   xy=(0,0), xytext=(pca.components_[D1,cIdx]*loadingVectorsScale, pca.components_[D0,cIdx]*loadingVectorsScale), fontsize=fontSize, arrowprops=dict(arrowstyle='<-', alpha=1.0, linewidth=1.5, color=colors[i]), color=colors[i], zorder=50 )

                ax4.annotate( u"\u0394LFE[{}nt]".format(abs(c)*10),   xy=(tlx-imw*1.9, bry-imh*2.2*(i+4)), xytext=(tlx-imw*0.4, bry-imh*2.2*(i+4)),  fontsize=fontSize, arrowprops=dict(arrowstyle='<-', alpha=1.0, linewidth=1.5, color=colors[i]), zorder=200 )

                

        #plt.scatter(X_reduced[:,1], X_reduced[:,0] )


        #plt.xlim( (min(X_reduced[:,D1]) - scaleX*0.1, max(X_reduced[:,D1]) + scaleX*0.1) )
        #plt.ylim( (min(X_reduced[:,D0]) - scaleY*0.1, max(X_reduced[:,D0]) + scaleY*0.1) )

        if layerConfig.showDists:
            sns.set(style="ticks")

            sns.distplot( X_reduced[:,D1].flatten(), hist=False, rug=False, ax=ax2 )
            sns.distplot( X_reduced[:,D0].flatten(), hist=False, rug=False, ax=ax3, vertical=True )


        if layerConfig.showDensity:
            ax4.set_autoscale_on(False)
            sns.kdeplot( X_reduced[:,D1].flatten(), X_reduced[:,D0].flatten(), n_levels=20, cmap="Blues", shade=True, shade_lowest=False, legend=False, ax=ax4, zorder=1 )
            #sns.jointplot( X_reduced[:,D1].flatten(), X_reduced[:,D0].flatten(), cmap="Blues" )
            # ax4.scatter(X_reduced[:,1], X_reduced[:,0], s=1.5, alpha=0.5 )


            
        if layerConfig.showTrait:
            xs = []
            ys = []
            cs = []
            for i, taxId in enumerate(biasProfiles.keys()):
                # Determine the location for this profile
                xs.append( X_reduced[i,D1] )
                ys.append( X_reduced[i,D0] )
                cs.append( traitValues.get(taxId, None) )
                
            traitPlot = ax4.scatter(xs, ys, c=cs, s=symbolScale, alpha=1.0, edgecolors='none', cmap="viridis", label="Trait"  )
            fig.colorbar(traitPlot, shrink=0.5)
            

        if layerConfig.debug:
            ax4.scatter(debugSymbols[0], debugSymbols[1], s=50, alpha=0.8, c="green", marker="+", zorder=300 )
            ax4.annotate( "*", xy=(D1_peak[0], D0_peak[0]), alpha=0.5, color='red', zorder=250 )

        ax4.set_ylabel('PCV1')
        ax4.set_xlabel('PCV2')


        #-----------------------------------------------------------------------------------
        fig.subplots_adjust(hspace=0, wspace=0, left=0.05, right=0.95, bottom=0.05, top=0.95)
        #plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        for ax in fig.axes:
            for side in ('top', 'right', 'left', 'bottom'):
                ax.spines[side].set_visible(False)
        ax1.set_xticks(())
        ax1.set_yticks(())

        ax4.set_aspect("equal")
        centerY = (max(X_reduced[:,D0]) + min(X_reduced[:,D0])) * 0.5
        centerX = (max(X_reduced[:,D1]) + min(X_reduced[:,D1])) * 0.5
        x0 = centerX - scaleX*0.55
        x1 = centerX + scaleX*0.55
        y0 = centerY - scaleY*0.55
        y1 = centerY + scaleY*0.55
        #ax4.set_xlim(( x0, x1 ))
        #ax4.set_ylim(( y0, y1 ))
        unifiedScaleForDistPlots = max(distPlotsScales)
        ax1.axis((unifiedScaleForDistPlots,  0,  0, unifiedScaleForDistPlots))
        ax2.axis((                      x0, x1,  0, unifiedScaleForDistPlots))
        ax3.axis((unifiedScaleForDistPlots,  0, y0, y1                      ))
        ax4.axis((                      x0, x1, y0, y1                      ))

        ax4.set_axis_on()
        plt.grid(False)

        if layerConfig.showTickMarks:
            ax4.set_xticks((min(X_reduced[:,D1]), max(X_reduced[:,D1])))
            ax4.set_yticks((min(X_reduced[:,D0]), max(X_reduced[:,D0])))
        else:
            ax4.set_xticks(())
            ax4.set_yticks(())

        if layerConfig.showDists:
            #ax2.set_yticks((0, round(unifiedScaleForDistPlots,2)))
            #Xax2.spines['left'].set_visible(True)
            #ax2.spines['right'].set_visible(True)
            
            #ax3.set_xticks((0, round(unifiedScaleForDistPlots,2)))
            #ax3.spines['bottom'].set_visible(True)
            #ax3.spines['top'].set_visible(True)
            pass
        
        #ax2.set_ylim((                 0, distPlotsScales[0]))
        #ax3.set_xlim((distPlotsScales[1],                  0))
        #ax3.invert_xaxis()
        #-----------------------------------------------------------------------------------

        plt.savefig("{}.pdf".format(layerConfig.output))
        plt.savefig("{}.png".format(layerConfig.output), dpi=300, transparent=True, bbox_inches='tight')
        plt.savefig("{}.svg".format(layerConfig.output))
        plt.close(fig)


    #_defaults = dict(showAxes=False, showDensity=False, showDists=False, showProfiles=False, showHighlights=False, showComponents=False, showLoadingVectors=False )
    plotPCALayer(LayerConfig(output="pca_profiles",         showAxes=True,  showDensity=False, showDists=True, showProfiles=True, showHighlights=False, showComponents=True, showLoadingVectors=True ) )
    plotPCALayer(LayerConfig(output="pca_profiles_density", showDensity=True ) )
    overlayImages( ["pca_profiles_density.png", "pca_profiles.png"], "pca_profiles_combined.png" )


    plotPCALayer(LayerConfig(output="pca_profiles_d",         showAxes=True,  showDensity=False, showDists=True, showProfiles=True, showHighlights=False, showComponents=True, showLoadingVectors=True, debug=True ) )
    plotPCALayer(LayerConfig(output="pca_profiles_density_d", showDensity=True, debug=True ) )
    overlayImages( ["pca_profiles_density_d.png", "pca_profiles_d.png"], "pca_profiles_combined_d.png" )
    

    plotPCALayer(LayerConfig(output="pca_profiles_trait",   showTrait=True, showLoadingVectors=True ) )
    overlayImages( ["pca_profiles_trait.png"], "pca_profiles_trait_combined.png" )

    
    print("Explained variance: {}".format(pca.explained_variance_ratio_))
    print("            (Total: {})".format(sum(pca.explained_variance_ratio_)))
    print("Components: {}".format(pca.components_[:2,:]))
    
    a = list(zip(list(biasProfiles.keys()), X_reduced[:,D0]))
    a.sort(key=lambda x:x[1])
    print("top:")
    print(a[:3])
    print("bottom:")
    print(a[-3:])
    return a


#def drawPCAforTree( profilesAsMap, xdata ):
#    arbitraryProfile =  profilesAsMap.values()[0]
#    assert(arbitraryProfile.ndim == 1)
#    profileLength = arbitraryProfile.shape[0]
#    numProfiles = len(profilesAsMap)#
#
#    profilesArray = np.zeros((numProfiles, profileLength))
#    for i, profile in enumerate(profilesAsMap.values()):
#        profilesArray[i] = profile#
#
#    PCAForProfiles( profilesArray, xdata )

def plotHistogramComparison( taxId, groups, labels ):
    fig, ax1 = plt.subplots()
    colors=("red","blue","yellow","black")

    bins = np.linspace( -15, 15, 51 )

    for i, data in enumerate(groups):
        ax1.hist( data, bins=bins, histtype='step', color=colors[i], label=labels[i] )

    
    plt.xlabel( "PA ratio")
    plt.ylabel( "Frequency" )
    speciesName = getSpeciesName(taxId)

    plt.title(speciesName)
    plt.legend(loc="upper left")
    
    baseName = "pa_ratios_{}".format( taxId )
    plt.savefig("{}.pdf".format( baseName ))
    plt.savefig("{}.svg".format( baseName ))
    plt.close(fig)
    

def plotPAratioVsDistanceHeatmap( taxId, data ):
    heatmapData = pd.DataFrame( { 'flankDistance':pd.Series(dtype='float'), 'logPAratio':pd.Series(dtype='float'), 'peakLFE':pd.Series(dtype='float')} )
    for flankingDistance in np.linspace( -50, 250, 11 ):
        for logPAratio in np.linspace( -2, 2, 9 ):
            meanPeakDLFE = data[(data['3utr_length'].between(flankingDistance,flankingDistance+30-1)) & (data['PAratio'].between(logPAratio,logPAratio+0.5-1e-6))]['mean_dLFE_for_peak'].mean()
            heatmapData = heatmapData.append( pd.DataFrame( { 'flankDistance':pd.Series([flankingDistance]), 'logPAratio':pd.Series([logPAratio]), 'peakDLFE':pd.Series([meanPeakDLFE]) } ))

    
    fig, ax = plt.subplots()
    
    sns.heatmap( heatmapData.pivot('flankDistance', 'logPAratio', 'peakDLFE'), ax=ax )
    
    
    plt.xlabel( "3'UTR distance")
    plt.ylabel( "log(PA ratio)" )
    speciesName = getSpeciesName(taxId)

    plt.title(speciesName)
    
    baseName = "pa_distance_heatmap_{}".format( taxId )
    plt.savefig("{}.pdf".format( baseName ))
    plt.savefig("{}.svg".format( baseName ))
    plt.close(fig)

    fig, ax = plt.subplots()
    sns.scatterplot( x='3utr_length', y='PAratio', hue='mean_dLFE_for_peak', size=2, alpha=0.5, data=data, ax=ax)
    plt.title(speciesName)
    plt.xlim([-50,250])
    plt.ylim([-3,3])
    
    baseName = "pa_distance_scatter_{}".format( taxId )
    plt.savefig("{}.pdf".format( baseName ))
    plt.savefig("{}.svg".format( baseName ))
    plt.close(fig)

    
def plotPAregression( taxId, data ):
    speciesName = getSpeciesName(taxId)

    fig, ax = plt.subplots()
    sns.lmplot( x='mean_dLFE_for_peak', y='PAratio', hue='IsNear', scatter_kws = {'alpha':0.3, 's':1.9}, fit_reg=True, data=data )
    
    plt.title(speciesName)
    
    baseName = "pa_regression_{}".format( taxId )
    plt.savefig("{}.pdf".format( baseName ))
    plt.savefig("{}.svg".format( baseName ))
    plt.close(fig)
    
