import csv
from genome_model import getGenomeModelFromCache
#import re

# configuration
ODB4_csv_path = "/tamir1/mich1/termfold/data/ODB4/known_operon.download.txt"

class ODB4_format(object):
    OperonId    = 0
    TaxId       = 1
    OperonName  = 2
    GenesOrder  = 3
    Description = 4
    Source_PMID = 5


def allItemsAreEqual(xs):
    assert(len(xs)>0)
    if len(xs)==1:
        return True
    else:
        return all([x==xs[0] for x in xs[1:]])

class ODB4(object):

    def __init__(self, taxIdFilter=None):
        self.taxIdFilter = taxIdFilter
        self.readFile()

    def getGeneIdentifier(self, geneId, taxId=None): #operonName, positionInOperon, numGenesInOperon, 
        if not taxId is None:
            return "{}:{}".format(geneId, taxId)
        else:
            assert(not self.taxIdFilter is None)
            return geneId
            

    def readFile(self):

        self.geneInfo = {}

        with open(ODB4_csv_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            rowNum = 0
            for row in reader:
                rowNum += 1
                if rowNum==1: continue # skip header line

                taxId = int(row[ODB4_format.TaxId])
                if (not self.taxIdFilter is None) and (taxId != self.taxIdFilter): continue  # filter rows by taxId (if specified)

                gm = getGenomeModelFromCache(taxId)

                operonName = row[ODB4_format.OperonName]
                
                genesInOrder = row[ODB4_format.GenesOrder].split(',')

                # For some reason, the genes in ODB4 appear relative to the positive strand, even for operons transcribed from the negative strand...
                # Consequently, to determine the real order of genes we need to determine the strand for this operon.
                # First, collect the strands for all genes in this operon
                strands = []
                for gid in genesInOrder: # for each gene, find equivalent identifiers
                    idents = gm.findEquivalentIdentifiers(gid) # try each identifier to find one associated with a gff3 feature
                    if idents is None:
                        if gid[:2]=="HP" and gid[:3]!="HP_":  # Workaround for H. pylori
                            idents = gm.findEquivalentIdentifiers("HP_"+gid[2:])
                            
                        if idents is None:
                            continue
                    
                    for i in idents:
                        f = gm.findFeatureById(i)
                        if not f is None:
                            strands.append( f[1].data['strand'] ) # store all strands
                            break

                if not strands: # With no strand found, we cannot use this operon
                    print("Missing info for {}".format(operonName))
                    continue

                # Make sure all strands are the same
                if not allItemsAreEqual(strands):
                    print("Conflicting info for {}".format(operonName))
                    continue
                strand = strands[0]
                assert(strand in ('+','-'))

                # Store position information for every gene in this operon
                for pos, gene in enumerate(genesInOrder):
                    if gene[:2]=="HP" and gene[:3]!="HP_":  # Workaround for H. pylori
                        gene="HP_"+gene[2:]
                        
                    if strand=='+':
                        geneData = (pos,                         len(genesInOrder), operonName)
                    else:
                        if taxId != 169963:
                            geneData = (len(genesInOrder) - pos - 1, len(genesInOrder), operonName)
                        else:  # Workaround for Listeria
                            geneData = (pos,                         len(genesInOrder), operonName)
                        
                    
                    if self.taxIdFilter is None: # No filter defined; store the taxId with the entry
                        self.geneInfo[self.getGeneIdentifier(gene, taxId=taxId)] = geneData
                    else: # Taxonomy filtered to single species; no need to store taxId with entry
                        self.geneInfo[self.getGeneIdentifier(gene)] = geneData
                        
    def findGeneData(self, geneId, taxId=None):
        return self.geneInfo.get( self.getGeneIdentifier( geneId, taxId=taxId ), None )


class OBD4Cache(object):
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
        return ODB4(taxIdFilter = taxId)

_obd4_cache = OBD4Cache()

def getOBD4FromCache( taxId ):
    return _obd4_cache[taxId]
    

def test():
    a = getOBD4FromCache(511145)
    genesToTest = ['b2801','b2802','b2803']
    for geneId in genesToTest:
        print("{} -> {}".format( geneId, a.findGeneData(geneId) ))

if __name__=="__main__":
    import sys
    test()


#gbfile = './data/NCBI/Pprofundum/NC_006370.1.gb'
#records = SeqIO.index_db(":memory:", gbfile, "genbank")
#records['NC_006370.1'].features[0]
#SeqFeature(FeatureLocation(ExactPosition(0), ExactPosition(4085304), strand=1), type='source')


#records['NC_006370.1'].features[1001].qualifiers['locus_tag']
#records['NC_006370.1'].features[1001].location.start
#records['NC_006370.1'].features[1001].location.end
#records['NC_006370.1'].features[1001].location.strand  # 1/-1




