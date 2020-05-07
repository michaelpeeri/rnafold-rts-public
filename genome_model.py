from collections.abc import Sequence
from typing import Optional
import json
from random import sample
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from Bio import SeqIO
import pyfaidx
from intervaltree import IntervalTree, Interval
from gff import createGffDb



_pyfaidxCache = {}
def pyfaidxCache(fastafn:str) -> object:
    ret = _pyfaidxCache.get(fastafn, None)
    
    if ret is None:
        newfaidx = pyfaidx.Fasta(fastafn)
        _pyfaidxCache[fastafn] = newfaidx
        ret = newfaidx
        
    return ret


_gbCache = {}
def genbankCache(genbankfn:str) -> object:
    ret = _gbCache.get(genbankfn, None)

    if ret is None:
        newgb = SeqIO.index_db(":memory:", genbankfn, "genbank")
        _gbCache[genbankfn] = newgb
        ret = newgb

    return ret


def allStopCodonsPositionsSource(aaseq:str):
    start = 0
    pos = aaseq.find('*')
    while pos >= 0:
        yield pos
        start = pos+1
        pos = aaseq.find('*', start)
    
def isValidCodingSequence(cdsNtSeq:str, geneticCode:int) -> bool:
    if not isinstance(cdsNtSeq, str):
        cdsNtSeq = str(cdsNtSeq).lower()
        
    cdsAASeq     = str(Seq(cdsNtSeq, generic_dna).translate(table=geneticCode)).lower()
    
    if (cdsNtSeq[-3:].find("n") == -1) and (cdsAASeq[-1] != "*"):  # the CDS must end with a stop-codon (or unknown nucleotides)
        return False # Invalid CDS
    
    if cdsAASeq[:-1].find('*')!=-1: # mid-CDS stop codon(s) found

        for internalStopCodonPos in allStopCodonsPositionsSource(cdsAASeq[:-1]):
            ntpos = internalStopCodonPos*3
            internalStopCodon = cdsNtSeq[ntpos:ntpos+3]
            if internalStopCodon == 'tga':
                print("Found '*' at pos {}, {}nt, {}".format(internalStopCodonPos, ntpos, internalStopCodon))
                # Internal UGA codon found; assume this is a selenocisteine site (we have no way of knowing)
            else:
                return False # non-UGA internal stop-codon found - this is not valid

        return True # if all internal stop-codons are UGAs, we will assume this CDS is ok

    return True # valid CDS
        

class IdentifiersConverter(object):
    def __init__(self):
        self._ids = {}

    def _addUnidirectionalMapping(self, id1, id2):
        if id1 in self._ids:
            self._ids[id1].append( id2 )
        else:
            self._ids[id1] = [id2]

    def addEquivalentPair(self, id1, id2):
        if id1 is None: return
        if id2 is None: return
        
        self._addUnidirectionalMapping( id1, id2 )
        self._addUnidirectionalMapping( id2, id1 )

    def findEquivalentIdentifiers(self, id_):
        return self._ids.get( id_, None)
        

trueVals = frozenset(('true', 'True', 'TRUE', '1'))
class GenomeMoleculeModel(object):

    _gf_molecule = 1
    _gf_start    = 4
    _gf_end      = 5
    _gf_strand   = 7
    _gf_props    = 9
    _debug_arrows = {'+':'----->', '-':'<-----'}
    
    def __init__(self, moleculeId:int, sequenceFile:str, gff, name:str, isLinear=None, props:dict ={}, wholeGenomeModel:object =None, genbankFile:Optional[str] = None):
        self.sequenceFile = sequenceFile
        self.fastaidx = pyfaidxCache(sequenceFile)
        if type(gff)==type(""):
            self.gffFile = gff
            # TODO load file
        else:
            self.gff = gff
        
        self.name = name
        self.props = props
        self.moleculeId = moleculeId
        
        if not isLinear is None: # get isLinear from props; allow isLinear to override
            self.isLinear = isLinear
        else:
            val = self.props.get('Is_circular', ['false'])
            self.isLinear = val[0] in trueVals

            
        self.features = IntervalTree()
        self.wholeGenomeModel = wholeGenomeModel
        self._identifiers = wholeGenomeModel.getIdentifiersConverter()
        
        self._loadFeatures()
        
        if not genbankFile is None:
            self._addGenbankLocusTags( genbankFile )


    """
    Select features from the gff3 file to be represnted internally, either directly as intervals in the interval tree or as properties of those intervals
    """
    def _loadFeatures(self):
        #print("mRNAs: {}".format( sum( 1 for _ in self.gff.region( seqid=self.name, featuretype='mRNA' ))))

        for geneft in self.gff.region( seqid=self.name, featuretype='gene' ):

            gene = geneft.astuple()
            geneprops = json.loads( gene[GenomeMoleculeModel._gf_props] )
            if "protein_coding" not in geneprops["biotype"]: continue

            # Get the 1st CDS
            fts = list( self.gff.children( geneft, featuretype="CDS" ) )
            if not fts: continue # gene features does not have CDS children
            feature = fts[0]
            ft = feature.astuple()
            assert( ft[GenomeMoleculeModel._gf_molecule]==self.name )

            # Get all exons
            exons = [x.astuple() for x in self.gff.children( geneft, featuretype="exon" )]
            assert( len(exons) >= 1)

            # ('transcript:ENST00000641515', '1', 'havana', 'mRNA', 65419, 71585, '.', '+', '.', '{"ID":["transcript:ENST00000641515"],"Parent":["gene:ENSG00000186092"],"Name":["OR4F5-202"],"biotype":["protein_coding"],"tag":["basic"],"transcript_id":["ENST00000641515"],"version":["2"]}', '[]', 4681)
            # ('transcript:ENST00000335137', '1', 'ensembl', 'mRNA', 69055, 70108, '.', '+', '.', '{"ID":["transcript:ENST00000335137"],"Parent":["gene:ENSG00000186092"],"Name":["OR4F5-201"],"biotype":["protein_coding"],"ccdsid":["CCDS30547.1"],"tag":["basic"],"transcript_id":["ENST00000335137"],"transcript_support_level":["NA (assigned to previous version 3)"],"version":["4"]}', '[]', 4681)
            # ('transcript:ENST00000426406', '1', 'ensembl_havana', 'mRNA', 450703, 451697, '.', '-', '.', '{"ID":["transcript:ENST00000426406"],"Parent":["gene:ENSG00000284733"],"Name":["OR4F29-201"],"biotype":["protein_coding"],"ccdsid":["CCDS72675.1"],"tag":["basic"],"transcript_id":["ENST00000426406"],"transcript_support_level":["NA (assigned to previous version 2)"],"version":["3"]}', '[]', 4684)
            # ('transcript:ENST00000332831', '1', 'ensembl_havana', 'mRNA', 685679, 686673, '.', '-', '.', '{"ID":["transcript:ENST00000332831"],"Parent":["gene:ENSG00000284662"],"Name":["OR4F16-201"],"biotype":["protein_coding"],"ccdsid":["CCDS41221.1"],"tag":["basic"],"transcript_id":["ENST00000332831"],"transcript_support_level":["NA (assigned to previous version 3)"],"version":["4"]}', '[]', 4686)

            sprops = ft[GenomeMoleculeModel._gf_props] 
            props = json.loads( sprops )
            strand = ft[GenomeMoleculeModel._gf_strand]
            assert( strand=='+' or strand=='-' )
            geneName = geneprops.get("Name", None)
            geneId = geneft.id
            data = {'strand':ft[GenomeMoleculeModel._gf_strand], 'props':sprops, 'gene-name':geneName, 'exons':exons, 'gene-id':geneId }
            # Use the gene boundaries (spanning all exons) for the interval
            start = gene[GenomeMoleculeModel._gf_start]
            end = gene[GenomeMoleculeModel._gf_end]

            # Extract the protId
            #print(props)
            assert(end >= start)
            protId = props["protein_id"][0]
            assert(len(protId)>3)

            # Store equivalent identifiers
            if isinstance(geneName, Sequence) and not isinstance(geneName, type("")):
                geneName = geneName[0] # take the first item
            if geneId[:5]=="gene:":
                geneId = geneId[5:]  # remove the 'gene:' prefix
            self._identifiers.addEquivalentPair( geneName, geneId )
            self._identifiers.addEquivalentPair( geneId, protId )
            self._identifiers.addEquivalentPair( protId, geneName )

            newinterval = Interval( start, end+1, data )  # end is inclusive in GFF3 but not inclusive in intervaltree
            self.features.add( newinterval )

            if not self.wholeGenomeModel is None:
                self.wholeGenomeModel.addFeatureIds( self.moleculeId, newinterval, (protId,) )

    def _addGenbankLocusTags( self, genbankFile:str ) -> None:
        gb = genbankCache( genbankFile )
        for key, record in gb.items():
            for feature in record.features:
                if feature.type != 'CDS': continue

                # Ensembl used "old" locus tags as gene names; ODB4 uses "new" locus tags.
                # We use genbank to map between the two
                if 'locus_tag' in feature.qualifiers and 'old_locus_tag' in feature.qualifiers:
                    locus_tag = feature.qualifiers['locus_tag'][0]
                    old_locus_tag = feature.qualifiers['old_locus_tag'][0]

                    matching = self._identifiers.findEquivalentIdentifiers(old_locus_tag)

                    if matching:
                        self._identifiers.addEquivalentPair( old_locus_tag, locus_tag )
    
    def findFeatureByStart( self, start:int ):
        val = self.features[start:start+1]
        if val:
            return list(val)[0]
        else:
            return None
        
    def findNextFeature(self, start:int, debug:bool=False):
        # TODO handle circular chromosomes
        width = 100
        while (True):

            candidates = sorted( self.features[start:start+width] )

            if debug:
                print("Trying with width={} ({}->{})...".format(width, start, start+width))
                print("candidates: {}".format(candidates))

            if candidates:
                return candidates[0]

            width *= 5
            if width > 2000000:
                return None
            
    def findPrevFeature(self, start:int, debug:bool=False):
        # TODO handle circular chromosomes
        width = 100
        while (True):
            
            candidates = sorted( self.features[start-width:start] )

            if debug:
                print("Trying with width={} ({}<-{})...".format(width, start-width, start))
                print("candidates: {}".format(candidates))

            if candidates:
                return candidates[-1]

            width *= 5
            if width > 2000000:
                return None

    def findFeatureById( self, featureId ):
        f = self.featureIds.get(featureId, None)
        if not f is None: return f

        for i in self._identifiers.findEquivalentIdentifiers( featureId ):
            f = self.featureIds.get(featureId, None)
            if f is None: continue
            if not f.moleculeId == self.moleculeId: continue

            return f
        return None

    def find3PrimeFlankingRegion( self, feature, debug:bool =False ):

        arrow=GenomeMoleculeModel._debug_arrows
        
        if feature.data['strand']=='+':
            next = self.findNextFeature( feature.end, debug=debug )
            if next is None:
                return None

            # This must hold, but, because of overalapping ORFs,  only approximately: ( feature.begin < feature.end <= next.begin < next.end )
            assert( feature.begin < feature.end )
            assert( next.begin < next.end )
            #print( next.begin  - feature.end )
            #assert( feature.end - 500 <= next.begin )
            #if not ( feature.end - 10 <= next.begin ):
            #    print("uig+")

            
            if debug:
                print("+")
                print("[[{}{}{}]]    [{}{}{}]".format( feature.begin, arrow[feature.data['strand']], feature.end, next.begin, arrow[next.data['strand']], next.end ))

                
            #ret = (self.name,                        feature.end, next.begin)
            ret = { "molecule":self.name,
                    "curr-feature-start": feature.begin,
                    "curr-feature": feature,
                    "curr-feature-end": feature.end,
                    "downstream-feature-start": next.begin,
                    "downstream-feature": next,
                    "downstream-feature-end": next.end,
                    "last-nucleotide": feature.end,
                    "flanking-region-start": feature.end,
                    "flanking-region-end": next.begin }
            
        elif feature.data['strand']=='-':
            prev = self.findPrevFeature( feature.begin-1, debug=debug )
            if prev is None:
                return None

            # This must hold, but, because of overalapping ORFs,  only approximately: ( prev.begin < prev.end <= feature.begin < feature.end )
            assert( prev.begin < prev.end )
            assert( feature.begin < feature.end )
            #print( feature.begin - prev.end )
            #assert( prev.end - 500 <= feature.begin )
            #if not ( prev.end - 10 <= feature.begin ):
            #    print("uig-")
            
            if debug:
                print("-")
                print("[{}{}{}]    [[{}{}{}]]".format( prev.begin, arrow[prev.data['strand']], prev.end, feature.begin, arrow[feature.data['strand']], feature.end ))
                
            #ret = (self.name, prev.end, feature.begin)
            ret = { "molecule":self.name,
                    "curr-feature-start": feature.begin,
                    "curr-feature": feature,
                    "curr-feature-end": feature.end,
                    "downstream-feature-start": prev.begin,
                    "downstream-feature": prev,
                    "downstream-feature-end": prev.end,
                    "flanking-region-start": prev.end,
                    "flanking-region-end": feature.begin }
            
        else:
            raise ValueError("Unknown strand '{}".format(feature.data['strand']))

        if debug:
            print(ret)
            
        #if ( ret[2] <= ret[1] ):
        #    return None
        
        return ret

    def find5PrimeFlankingRegion( self, feature, debug:bool =False ):
        
        if feature.data['strand']=='+':
            prev = self.findPrevFeature( feature.begin-1, debug=debug )
            if prev is None:
                return None
            
            ret = { "molecule":self.name,
                    "curr-feature-start": feature.begin,
                    "curr-feature": feature,
                    "curr-feature-end": feature.end,
                    "downstream-feature-start": prev.begin,
                    "downstream-feature": prev,
                    "downstream-feature-end": prev.end,
                    "last-nucleotide": feature.end,
                    "flanking-region-start": feature.end,
                    "flanking-region-end": prev.begin }
            

        elif feature.data['strand']=='-':
            next = self.findNextFeature( feature.end, debug=debug )
            if next is None:
                return None
            
            ret = { "molecule":self.name,
                    "curr-feature-start": feature.begin,
                    "curr-feature": feature,
                    "curr-feature-end": feature.end,
                    "downstream-feature-start": next.begin,
                    "downstream-feature": next,
                    "downstream-feature-end": next.end,
                    "flanking-region-start": next.end,
                    "flanking-region-end": feature.begin }

        else:
            raise ValueError("Unknown strand '{}".format(feature.data['strand']))

        return ret
        
            
    def numGenes(self) -> int:
        return len(self.features)

    # Get a nucleotide range
    def getSequence(self, begin, end, rc=False):
        return self.fastaidx.get_seq(self.name, begin, end, rc=rc)

    def getFeatureSplicedmRNA(self, feature):

        if isinstance(feature, str):
            feature = self.findFeatureById(feature)

        exonSeqs = []
        for exon in feature.data['exons']:
            exonStart = exon[GenomeMoleculeModel._gf_start]
            exonEnd   = exon[GenomeMoleculeModel._gf_end]

            exonSeqs.append( self.getSequence( exonStart, exonEnd, rc=(True if feature.data['strand']=='-' else False) ).seq )
        cdsSeq = Seq( "".join( exonSeqs ), generic_dna )
        
        return cdsSeq
        

class GenomeModel(object):

    _gf_mol_name  = 1
    _gf_mol_props = 9

    def __init__(self, sequenceFile:str, gffFile:str, isLinear =None, variant:str ="Ensembl", geneticCode:int =1, genbankFile:Optional[str] =None):
        self.sequenceFile = sequenceFile
        self.gffFile = gffFile
        self.genbankFile = genbankFile
        self.variant = variant
        self.geneticCode = geneticCode

        print("Loading gff3 file...")
        self.gff = createGffDb(gffFile, variant)
        print("Done.")

        chromosomeFeatures = list(self.gff.features_of_type(("chromosome", "supercontig" )) )
        moleculeNames = [c.astuple()[GenomeModel._gf_mol_name] for c in chromosomeFeatures]
        moleculeProps = [json.loads(c.astuple()[GenomeModel._gf_mol_props]) for c in chromosomeFeatures]
        #self.molecules = [Genome(name,props) for (name,props) in zip(moleculeNames, moleculeProps)]
        #assert(self.molecules)
        #print( [json.loads(c.astuple()[9]) for c in self.gff.features_of_type( "chromosome" )] )

        self.featureIds = {}

        self._identifiersConverter = IdentifiersConverter()

        self.moleculeModels = [GenomeMoleculeModel(moleculeId=molId,
                                                   sequenceFile=sequenceFile,
                                                   gff=self.gff,
                                                   name=name,
                                                   isLinear=isLinear,
                                                   props=props,
                                                   wholeGenomeModel=self,
                                                   genbankFile=genbankFile) for (molId,name,props) in zip(range(len(moleculeNames)), moleculeNames, moleculeProps)]

        print([x.name for x in self.moleculeModels])
        print([len(x.features) for x in self.moleculeModels])

        self.fasta = pyfaidxCache(sequenceFile)

    def getRegion(self, region:tuple, rc=False):
        (chromosome, begin, end) = region
        return self.fasta.get_seq(chromosome, begin, end, rc=rc)

    """
    Maintain a dictionary holding identifiers to all features, to allow lookup by id
    """
    def addFeatureIds( self, moleculeId, feature, featureIds ) -> None:
        for identifier in featureIds:
            self.featureIds[identifier] = (moleculeId, feature) # (<index in molecules array>, <feature obj>)

    def findFeatureById( self, featureId ) -> (int, Interval):
        f = self.featureIds.get(featureId, None)
        if not f is None: return f

        for i in self._identifiersConverter.findEquivalentIdentifiers( featureId ):
            f = self.featureIds.get(i, None)
            if f is None: continue

            return f
        return None
        

    def allCDSSource(self):
        for mol in self.moleculeModels:
            for interval in mol.features:
                sprops = interval.data["props"]
                props = json.loads( sprops )

                yield props["protein_id"][0]

    def numGenes(self) -> int:
        return sum( [mol.numGenes() for mol in self.moleculeModels] )

    def getGeneticCode(self) -> int:
        return self.geneticCode

    def getIdentifiersConverter(self):
        return self._identifiersConverter

    def findEquivalentIdentifiers(self, id_):
        return self._identifiersConverter.findEquivalentIdentifiers(id_)
        
        

class GenomeModelsCache(object):
    def __init__(self):
        self._cache = {}

    def __getitem__(self, taxId):
        val = self._cache.get(taxId, None)

        if not val is None:
            return val
        else:
            val = self._init_item(taxId)
            self._cache[taxId] = val
            return val
        
    def _init_item(self, taxId):
        from data_helpers import getSpeciesGenomeSequenceFile, getSpeciesGenomeAnnotationsFile, getSpeciesGenomeAnnotationsVariant, getSpeciesTranslationTable, getSpeciesGenbankAnnotationsFile
        
        genomeSeqFile      = getSpeciesGenomeSequenceFile( taxId )
        genomeAnnotFile    = getSpeciesGenomeAnnotationsFile( taxId )
        genomeAnnotVariant = getSpeciesGenomeAnnotationsVariant( taxId )
        genomeGenbankFile  = getSpeciesGenbankAnnotationsFile( taxId )
        geneticCode        = getSpeciesTranslationTable( taxId )
        if genomeSeqFile is None or genomeAnnotFile is None or geneticCode is None:
             raise ValueError("No supporting annotations for taxId={}".format(taxId))

        gm = GenomeModel(
            sequenceFile=genomeSeqFile,
            gffFile=genomeAnnotFile,
            isLinear=False,   # TODO fix this
            variant=genomeAnnotVariant,
            geneticCode=geneticCode,
            genbankFile=genomeGenbankFile )

        return gm

_genome_models_cache = GenomeModelsCache()

def getGenomeModelFromCache( taxId ):
    return _genome_models_cache[taxId]

    
def displayInterval(tree, begin:int, end:int):
    print(tree[begin-10:end+11])

def testForwardIteration(mol):
    allFeatures = sorted( mol.features.items() )
    print(len(allFeatures))

    iteratedFeatures = []
    node = mol.findNextFeature(0)
    iteratedFeatures.append(node)
    
    while (True):
        node = mol.findNextFeature(node.end+1)
        
        if node is None:
            break
        
        iteratedFeatures.append(node)
        
    print(len(iteratedFeatures))

    missingFeatures = frozenset(allFeatures) - frozenset(iteratedFeatures)
    print(missingFeatures)
    for f in missingFeatures:
        print("--> missing:")
        displayInterval( mol.features, f.begin, f.end )
        

def testBackwardIteration(mol):
    allFeatures = list( reversed( sorted( mol.features.items() )))
    print(len(allFeatures))

    last = allFeatures[0]

    iteratedFeatures = []
    node = mol.findPrevFeature(last.end+100)
    iteratedFeatures.append(node)
    
    while (True):
        node = mol.findPrevFeature(node.begin)
        
        if node is None:
            break
        
        iteratedFeatures.append(node)
    print(len(iteratedFeatures))

    missingFeatures = frozenset(allFeatures) - frozenset(iteratedFeatures)
    print(missingFeatures)
    for f in missingFeatures:
        print("--> missing:")
        displayInterval( mol.features, f.begin, f.end )

def test3primeFlankingRegions(mol):
    from collections import Counter

    stats = Counter()

    successSet = set()
    failSet = set()
    over500set = set()
    over10set = set()
    over40set = set()
    over100set = set()
    over250set = set()
    over500set = set()
    
    for feat in mol.features.items():
        val = mol.find3PrimeFlankingRegion( feat )
        if val is None:
            failSet.add(feat)
            continue
        
        (chromosome, begin, end) = val
        assert( end > begin )
        stats.update( (end-begin,) )
        successSet.add(feat)
        
        if end-begin >= 10:
            over10set.add(feat)
            if end-begin >= 40:
                over40set.add(feat)
                if end-begin >= 100:
                    over100set.add(feat)
                    if end-begin >= 250:
                        over250set.add(feat)
                        if end-begin >= 500:
                            over500set.add(feat)

    print("Found {} flanking regions".format(sum(stats)))
    print(len(successSet))
    print(stats.most_common(10))

    print("Failed (sample):")
    for f in sample( failSet, 10 ):
        print(f)

    print("Over100 (sample):")
    for f in sample( over100set, 20 ):
        print(f)

        
    print("Count (len >  10): {}".format( len( over10set  )))
    print("Count (len >  40): {}".format( len( over40set  )))
    print("Count (len > 100): {}".format( len( over100set )))
    print("Count (len > 250): {}".format( len( over250set )))
    print("Count (len > 500): {}".format( len( over500set )))
    

    # Test cases:
    # Overlap over start codon
    print( mol.findFeatureByStart(550439) )
    mol.find3PrimeFlankingRegion( mol.findFeatureByStart(550439), debug=True )
    print(mol.findNextFeature(550493))

    print("---"*10)
    print(mol.features[59000:60000])
    print("---"*10)
    

    f59687 = mol.findFeatureByStart(59687)
    mol.find3PrimeFlankingRegion( f59687, debug=True )
    assert( f59687 in over100set )
    #assert( f59687 not in over500set )  - fails because some CDSs in the middle of this UTR are missing

# minLength==-1 -> no minimum length
def CDS3PrimeFlankingRegionSource( mol:object, minLength:int=-1, debug:bool=False ):
    for feat in mol.features.items():
        region = mol.find3PrimeFlankingRegion( feat, debug=debug )
        
        if not region is None:
            assert feat==region["curr-feature"]
            
            length = region["flanking-region-end"]-region["flanking-region-start"]
            if minLength < 0 or length >= minLength:
                yield( feat, region )

def CDS3PrimeFlankingRegionWithDownstreamFeatureSource( mol:object, debug:bool=False ):

    alreadyDone = set()
    
    for feat in mol.features.items():
        region = mol.find3PrimeFlankingRegion( feat, debug=debug )
        
        if not region is None:
            assert feat==region["curr-feature"]
            downstream = region["downstream-feature"]

            rprops = json.loads( feat.data['props'] )
            parentId = rprops['Parent'][0]
            if parentId in alreadyDone:
                raise Exception("Selected features with identical parent-ids from gff3 {}".format(parentId))
            else:
                alreadyDone.add( parentId )
            
            #length = region["flanking-region-end"]-region["flanking-region-start"]
            yield( feat, region )
    

def CDSWith3PrimeSequencesSource( genomeModel:object, minLength:int=10, debug:bool=False ):
    numRegions = 0
    
    for mol in genomeModel.moleculeModels:
        for feat, region in CDS3PrimeFlankingRegionSource( mol, minLength=minLength, debug=debug ):
            if debug:
                print("-----------"*3)
                print(feat)
                print(region)
            
            if feat.data['strand']=='+':
                #begin = region[1]-30
                begin = feat.begin
                end = region["flanking-region-end"]
                seq = genomeModel.getRegion( (region["molecule"], begin, end) )
            
            elif feat.data['strand']=='-':
                begin = region["flanking-region-start"]  -1
                #end = region[2]+30 -1
                end = feat.end -1
                seq = genomeModel.getRegion( (region["molecule"], begin, end), rc=True )

            else:
                raise ValueError("Unknown strand '{}".format(feature.data['strand']))

            #print(seq[:42])
            ntSeq = Seq( seq.seq, generic_dna )
            if debug:
                print(ntSeq)
                print(len(seq.seq))
            
            #if feat.data['strand']=='-':
            #    ntSeq = ntSeq.reverse_complement()

            aaSeq = ntSeq.translate(table=genomeModel.geneticCode)
            numRegions += 1

            stopCodonPos = feat.end - feat.begin - 3

            if debug:
                print(aaSeq)
                print(aaSeq[((stopCodonPos-6)//3):((stopCodonPos+9)//3):])

            aaAtStopCodonPosition = aaSeq[stopCodonPos//3]
            if not (aaAtStopCodonPosition=="*" or aaAtStopCodonPosition=="X"):
                print("Warning: stop codon not found...")
            
                
            yield (feat, region, ntSeq, stopCodonPos)
            


    print("Found {} regions".format(numRegions))



def CDSWith3PrimeSequencesWithDownstreamFeatureSource( genomeModel:object, debug:bool=False ): # polycistronic mRNA
    numRegions = 0
        #
    for mol in genomeModel.moleculeModels:
        for feat, region in CDS3PrimeFlankingRegionWithDownstreamFeatureSource( mol, debug=debug ):
            if debug:
                print("-----------"*3)
                print(feat)
                print(region)

            flanking = region['downstream-feature']
            
            if feat.data['strand']=='+':

                downstreamFeatureEnd = region["downstream-feature-end"]
                
                #begin = feat.begin
                #end = downstreamFeatureEnd

                gap = region["downstream-feature-start"] - feat.end
                utr3Start = feat.end
                utr3End = region["downstream-feature-start"]
                
                #seq = genomeModel.getRegion( (region["molecule"], begin, end) )
                if gap > 0:
                    utr3Seq = genomeModel.getRegion( (region["molecule"], utr3Start, utr3End-1) )
                else:
                    utr3Seq = Seq("", generic_dna)
                # TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #
                #utr3Seq = Seq("N" * len(utr3Seq), generic_dna )
                # TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #
            
            elif feat.data['strand']=='-':

                downstreamFeatureStart = region["downstream-feature-start"]
                
                #begin = downstreamFeatureStart
                #end = feat.end-1

                gap = feat.begin - region["downstream-feature-end"]
                utr3Start = region["downstream-feature-end"]
                utr3End = feat.begin
                
                #seq = genomeModel.getRegion( (region["molecule"], begin, end), rc=True )
                if gap > 0:
                    utr3Seq = genomeModel.getRegion( (region["molecule"], utr3Start, utr3End-1), rc=True )
                else:
                    utr3Seq = Seq("", generic_dna)
                    
                # TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #
                #utr3Seq = Seq("N" * len(utr3Seq), generic_dna )
                # TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #### TEST ONLY #

            else:
                raise ValueError("Unknown strand '{}".format(feature.data['strand']))

            # exonSeqs = []
            # for exon in feat.data['exons']:
            #     exonStart = exon[GenomeMoleculeModel._gf_start]
            #     exonEnd   = exon[GenomeMoleculeModel._gf_end]
                
            #     exonSeqs.append( genomeModel.getRegion( (region["molecule"], exonStart, exonEnd ), rc=(True if feat.data['strand']=='-' else False) ).seq )
            # cdsSeq = Seq( "".join( exonSeqs ), generic_dna )

            cdsSeq = mol.getFeatureSplicedmRNA( feat )
            nextCdsSeq = mol.getFeatureSplicedmRNA( region["downstream-feature"] )
            nextCdsStrand = region['downstream-feature'].data['strand']
            
            
            #print(seq[:42])
            #ntSeq = Seq( seq.seq, generic_dna )
            if debug:
                print(cdsSeq)
                print(len(seq.seq))
            
            #if feat.data['strand']=='-':
            #    ntSeq = ntSeq.reverse_complement()

            if( len(cdsSeq) % 3 != 0 ):
                #raise Exception("Coding sequence has partial codon (length={})".format( len(ntSeq) ))
                print("-------"*8)
                print("Warning: Coding sequence has partial codon (length={})".format( len(cdsSeq) ))
                print(feat)
                print("-------"*8)
                continue
            
            aaSeq = cdsSeq.translate(table=genomeModel.geneticCode)
            numRegions += 1

            #stopCodonPos = feat.end - feat.begin - 3
            stopCodonPos = len(cdsSeq)-3

            if debug:
                print(aaSeq)
                print(aaSeq[((stopCodonPos-6)//3):((stopCodonPos+9)//3):])

            if not ( gap>0 or len(utr3Seq)==0 ):
                print(gap)
            #assert( gap>0 or len(utr3Seq)==0 )  # if gap<=0, there is no UTR seq
            if not ( gap<0 or len(utr3Seq)==gap ):
                print(gap)
            #assert( gap<0 or len(utr3Seq)==gap )  # if gap>=0, the UTR seq has length==gap

            aaAtStopCodonPosition = aaSeq[stopCodonPos//3]
            if not (aaAtStopCodonPosition=="*" or aaAtStopCodonPosition=="X"):
                #print("Warning: stop codon not found...")
                #raise Exception("Stop codon not found...")
                print("-------"*8)
                print("Warning: Stop codon not found")
                print(feat)
                print("-------"*8)
                continue


            #nextStartCodonPos = 
                
            yield (feat, region, cdsSeq, utr3Seq, gap, nextCdsSeq, nextCdsStrand )
            


    print("Found {} regions".format(numRegions))


def writeFlankingSeqToFasta( genomeModel, minLength:int=10, debug=False ):
    #from Bio import SeqIO
    #from Bio.SeqRecord import SeqRecord
    #from Bio.Seq import Seq
    #from Bio.Alphabet import generic_dna

    numRegions = 0

    for (feat,region,ntSeq,stopCodonPos) in CDSWith3PrimeSequencesSource( genomeModel, minLength=minLength, debug=debug ):

            #print(seq[:42])
            if debug:
                print(ntSeq)
                print(len(ntSeq))
            
            #if feat.data['strand']=='-':
            #    ntSeq = ntSeq.reverse_complement()

            aaSeq = ntSeq.translate(table=genomeModel.geneticCode)
            numRegions += 1
            
            if debug:
                print(aaSeq)
            
            #recs.append( SeqRecord( Seq(seq), id=seq.name, name=seq.fancy_name, description=orig.description) )


    print("Found {} regions".format(numRegions))
    #with open(args.output_fasta, "w") as outfile:
    #    out = SeqIO.write( outRecords, outfile, "fasta")

        

def testAll():
    # gm1 = GenomeModel(
    #     sequenceFile='/tamir1/mich1/cellfold/data/Ensembl/Homo.sapiens/Homo_sapiens.GRCh38.dna_rm.toplevel.fa.gz',
    #     gffFile='/tamir1/mich1/cellfold/data/Ensembl/Homo.sapiens/Homo_sapiens.GRCh38.95.gff3.gz',
    #     isLinear=True,
    #     variant="Ensembl",
    #     geneticCode=1 )
                               
    gm2 = GenomeModel(
        sequenceFile='/tamir1/mich1/termfold/data/Ensembl/Ecoli/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.dna_rm.toplevel.fa', # bgzip compression is not supported yet!
        gffFile=     '/tamir1/mich1/termfold/data/Ensembl/Ecoli/Escherichia_coli_str_k_12_substr_mg1655.ASM584v2.37.gff3.gz',
        isLinear=True,
        variant="Ensembl",
        geneticCode=11 )

    print("Forward:")
    testForwardIteration(  gm2.moleculeModels[0] )
    print("Backward:")
    testBackwardIteration( gm2.moleculeModels[0] )

    test3primeFlankingRegions( gm2.moleculeModels[0] )

    #writeFlankingSeqToFasta( gm2, minLength=1, debug=True )
    writeFlankingSeqToFasta( gm2, minLength=1, debug=True )
        
    
    return 0

    

if __name__=="__main__":
    import sys
    sys.exit( testAll() )
