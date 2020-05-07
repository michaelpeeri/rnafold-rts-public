from __future__ import print_function
from builtins import str
from builtins import range
import sys
import subprocess
from tempfile import NamedTemporaryFile
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import NucleotideAlphabet
import config


# Configuration
codonwMainPath = "%s/codonw" % config.codonwBasePath
#codonwEncPath = "%s/enc" % codonwBasePath
#codonwCBIPath = "%s/cbi" % codonwBasePath
#codonwCAIPath = "%s/cai" % codonwBasePath
#codonwFopPath = "%s/fop" % codonwBasePath
#codonwCaiPath = "%s/cai" % codonwBasePath

def fastaSourceRaw(filename):
    for seq in SeqIO.parse(open(filename, 'r'), 'fasta'):
        yield seq


def translateFastaAlphabet(source, mapping):
    out = []

    # create translation table to be used by str.translate()
    s_from = ''.join(list(mapping.keys()))
    s_to = ''.join(list(mapping.values()))
    assert(len(s_from)==len(s_to))
    trans = str.maketrans(s_from, s_to)
    
    for orig in source:
        translatedSeq = str(orig.seq).translate(trans)
        out.append( SeqRecord( Seq(translatedSeq), id=orig.id, name=orig.name, description=orig.description))
        #out.append( SeqRecord( Seq(translatedSeq), id=orig.id, description="REMOVE" ) )

    outname = "/tmp/test.fna"
    SeqIO.write( out, outname, "fasta")
    return outname


def truncateAtFirstUnderscore(geneid):
    if( geneid.find('_') <= 3 ):
        return '_'.join(geneid.split('_')[:2])
    else:
        return geneid.split('_')[0]

"""
Run codonw to get gene-level CUB metrics
Input - path to fasta file containing sequences
Output - pandas dataframe containing the results
"""
def readCodonw(fastaPath):
    fout = NamedTemporaryFile(mode='r')
    
    #codonw /tmp/test.fna /tmp/test.out -all_indices -silent -nomenu
    out = subprocess.check_output((codonwMainPath, fastaPath, fout.name, "-all_indices", "-nomenu", "-silent"), shell=False)
    
    df = pd.read_csv(
        fout, sep='\t', index_col=0, na_values='*****',
        dtype={}, #'L_aa':np.int, 'L_sym':np.int},
        converters={0:truncateAtFirstUnderscore}
    )
    
    df = df[df['L_sym'].notnull()]  # some records are all null; drop these
    df['L_sym'] = df['L_sym'].astype(np.int) # after dropping null columns, the integral columns can be converted
    df['L_aa'] = df['L_aa'].astype(np.int)
    
    return df
        
"""
Run codonw to get a profile of CUB metrics (averaged over all genes, for each window)
Most metrics cannot be computed for a single-gene window, but we can compute  them over that window on the entire genome
Input - fullSeqs - container of SeqRecord objects
Output - dataframe
"""
def meanCodonwProfile(fullSeqs, maxWindowWidth=40, profileReference='begin', profileStart=0, profileStop=600, profileStep=10):
    assert(profileReference=='begin')

    #print("Profile: reference: %s start: %d end: %d step: %d window-size: %d" % (profileReference, profileStart, profileStop, profileStep, maxWindowWidth ))

    seqsForProfiles = []

    for start in range(profileStart, profileStop, profileStep):
        # Find a real window that falls on codon boundaries
        # We use the maximal fragment fully contained within the window
        # (so the actual fragment size might be up to 4 nucleotides shorter than the window - it can lose up to 2 from either side)
        # The real window will be the range [realStart, realStop)
        # realStart must be the first nucleotide in a codon
        # realEnd must be one past the last nucleotide in a codon
        realStart = start+((3-(start%3))%3)
        realStop = start+maxWindowWidth - ((start+maxWindowWidth)%3)

        #print((realStart,realStop))
        
        assert(realStart%3==0)  # Start must be the first nucleotide in a codon
        assert(realStop%3==0)   # End must be one past the last nucleotide in a codon
        assert((realStop-realStart)%3==0)
        assert((realStop-realStart) >= maxWindowWidth-4)

        # Collect this fragment for each sequence
        fragments = []
        for seq in fullSeqs: # each seq is a SeqRecord object
            assert(len(seq.seq) % 3 == 0)

            if( len(seq.seq) < realStop ):
                continue
            
            windowSeq = str(seq.seq[realStart:realStop])
            assert(len(windowSeq)%3==0)
            fragments.append(windowSeq)

        combinedSeq = ''.join(fragments)
        assert(len(combinedSeq)%3==0)

        seqsForProfiles.append( SeqRecord( Seq(combinedSeq, NucleotideAlphabet), id="profile-%d-%d-%d"%(start, realStart, realStop)) )

        

    fSeqsForProfiles = NamedTemporaryFile(mode="w")         # create a temporary file
    SeqIO.write( seqsForProfiles, fSeqsForProfiles.name, "fasta")  # write the full sequences into the file
    dfResults = readCodonw( fSeqsForProfiles.name )
    windowStarts = [int(x.split('-')[1]) for x in dfResults.index]
    dfResults['windowStart'] = pd.Series(windowStarts, index=dfResults.index)
    return dfResults
            
        
        
"""
"""
def translateFileForCodonW(filein):
    #fileout = translateFastaAlphabet(fastaSourceRaw(filein), {'t':'u', 'T':'U'} )
    fileout = translateFastaAlphabet(fastaSourceRaw(filein), {} )
    return fileout

def test():
    translated = translateFileForCodonW('./data/orf_coding.fasta')
    df = readCodonw(translated)
    print(df)
    print(df.corr())

    s = df['GC3s'].copy()
    df2 = pd.DataFrame(s)
    df2['rand'] = pd.Series( np.random.randn(len(df2)), index=df2.index )
    df2['GC3s_alt'] = pd.Series( df2['GC3s']*10 - 5 )
    df2.sort_values(by='rand', inplace=True)

    df3 = pd.merge(df, df2)
    print(df3.corr())
