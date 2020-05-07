from __future__ import print_function
from builtins import map
import argparse
import gff


#
# JGI format
#
# gff3:
#
#scaffold_1      phytozomev10    gene    796363  797228  .       -       .       ID=estExt_fgenesh1_pg.C_10151.3.0.228;Name=estExt_fgenesh1_pg.C_10151
#scaffold_1      phytozomev10    mRNA    796363  797228  .       -       .       ID=209109.3.0.228;Name=209109;pacid=27345144;longest=1;Parent=estExt_fgenesh1_pg.C_10151.3.0.228
#scaffold_1      phytozomev10    three_prime_UTR 796363  796397  .       -       .       ID=209109.3.0.228.three_prime_UTR.1;Parent=209109.3.0.228;pacid=27345144
#scaffold_1      phytozomev10    CDS     796398  797189  .       -       0       ID=209109.3.0.228.CDS.1;Parent=209109.3.0.228;pacid=27345144
#scaffold_1      phytozomev10    five_prime_UTR  797190  797228  .       -       .       ID=209109.3.0.228.five_prime_UTR.1;Parent=209109.3.0.228;pacid=27345144
#
# CDS fasta:
#>209109 pacid=27345144 polypeptide=209069 locus=estExt_fgenesh1_pg.C_10151 ID=209109.3.0.228 annot-version=v3.0
#

def parseOption(possibleValues, name):
    def checkOption(value):
        if value in possibleValues:
            return value
        else:
            raise argparse.ArgumentTypeError("Unknown %s '%s', allowed values: %s" % (name, value, ",".join(possibleValues)))
    return checkOption

def parseList(conversion=str):
    def convert(values):
        return list(map(conversion, values.split(",")))
    return convert

argsParser = argparse.ArgumentParser()
#argsParser.add_argument("--taxid", type=int)
argsParser.add_argument("--gff")
argsParser.add_argument("--variant", type=parseOption(set(("yeastgenome","NCBI","Ensembl")), "variant"))
argsParser.add_argument("--output-gene-ids")
argsParser.add_argument("--areas", type=parseList(int))
argsParser.add_argument("--transl-table", type=int)
argsParser.add_argument("--alt-protein-ids", type=parseOption(set(("locus_tag",)), "alt-protein-id"))
args = argsParser.parse_args()
assert(args.transl_table)

print("Loading gff file...")
db = gff.createGffDb(args.gff, args.variant)
print("Done.")


# Collect all Yeast coding genes; skip 'Dubious' ORFs
includedGenes = 0
skippedGenes = 0
duplicateCDSGenes = 0

codingGenes = set()

prev = 0


def getRegions():
    out = []
    if( args.variant=="yeastgenome.org" or args.variant=="NCBI"):
        for region in db.features_of_type("region"):
            out.append(region[0])
    elif( args.variant=="Ensembl"):
        out=[None]
    else:
        assert(False)
    return out

dbxref = set()

def processCodingGene(identifier, cds, parent):
    codingGenes.add(identifier)
    if 'Dbxref' in cds.attributes:
        dbxref.add(cds.attributes['Dbxref'][0])

allAreas = args.areas
if not allAreas:
    allAreas = getRegions()


def CDS_and_mRNA_source():
    if( args.variant=="yeastgenome.org" or args.variant=="NCBI"):
        for cds in db.features_of_type('CDS', limit=(area, 0, 100000000), completely_within=False):
            for mRNA in db.parents(cds.id):
                for gene in db.parents(mRNA.id):
                    yield cds, mRNA, gene

    elif( args.variant=="Ensembl"):
        for cds in db.features_of_type('CDS'):
            #print("CDS.id: %s" % cds.id)
            mRNA = None
            gene = None
            for p in db.parents(cds.id):
                #print("mRNA.featuretype: %s" % (mRNA.featuretype))
                if p.featuretype=='gene':
                    gene = p
                elif p.featuretype=='mRNA' or p.featuretype=='miRNA':
                    mRNA = p
                elif p.featuretype=='transcript':
                    mRNA = p
            yield cds, mRNA, gene

    else:
        assert(False)

for area in allAreas:
#for area in ('chrI', 'chrII', 'chrIII', 'chrIV', 'chrV', 'chrVI', 'chrVII', 'chrVIII', 'chrIX', 'chrX', 'chrXI', 'chrXII', 'chrXIII', 'chrXIV', 'chrXV', 'chrXVI'):     #, 'chrmt'):
    codingGenesInArea = 0
    for cds, mRNA, gene in CDS_and_mRNA_source():
        if(args.variant=="yeastgenome.org"):
            #for gene in db.parents(mRNA.id):
            if gene.featuretype == 'gene':
                if args.variant=="yeastgenome.org" and gene.attributes['orf_classification'][0] == 'Dubious':
                    skippedGenes += 1
                    continue

                if gene.id in codingGenes:
                    duplicateCDSGenes += 1
                #    print("%s already in!" % (gene.id,))
                # Note - manually tested that duplicates have identical coordinate
                processCodingGene(gene.id, cds, gene)
                codingGenesInArea += 1

        elif( args.variant=="NCBI"):
            # TODO - FIX THIS!!!!
            if(mRNA.featuretype != "gene"):
                skippedGenes += 1
                continue

            assert(mRNA.featuretype=="gene")

            #print(mRNA)

            # Skip pseudo-genes
            if( mRNA.attributes['gene_biotype'][0] == 'pseudogene' ):
                skippedGenes += 1
                continue

            if( 'pseudo' in cds.attributes and cds.attributes['pseudo'][0] == 'true' ):
                skippedGenes +=1
                continue

            if( 'transl_table' in cds.attributes and int(cds.attributes['transl_table'][0]) != args.transl_table ):
                skippedGenes += 1
                continue

            #if( mRNA.attributes['gene'][0][:3] == 'ins' ):
            #    print(cds)
            #    print(mRNA)

            if 'Dbxref' in mRNA.attributes:
                _Dbxrefs = mRNA.attributes['Dbxref']
            elif 'Dbxref' in cds.attributes:  # On some gff files, the attributes appear on the CDS feature...
                _Dbxrefs = cds.attributes['Dbxref']
            else:
                skippedGenes += 1
                continue
                
            assert(len(_Dbxrefs)>0)
            Dbxrefs = dict([tuple(x.split(":")) for x in _Dbxrefs])

            # Extract the gene name
            #geneId = Dbxrefs['GeneID']
            if not args.alt_protein_ids:
                geneId = cds.attributes['protein_id'][0]
            elif args.alt_protein_ids=='locus_tag':
                geneId = mRNA.attributes['locus_tag'][0]


            processCodingGene(geneId, cds, mRNA)
            if( mRNA.id in codingGenes ):
                print("Duplicate entries for %s" % (mRNA.id,))
                duplicateCDSGenes += 1
                    
                
                    

            codingGenesInArea += 1

        elif(args.variant=="Ensembl"):
            #print("-")
            biotype = mRNA.attributes['biotype'][0]
            if( biotype != 'protein_coding' ):
                skippedGenes += 1
                continue

            geneId = mRNA.attributes['transcript_id'][0]
            if( not geneId ):
                print("Missing gene-id!")
                assert(False)
                
            
            print(cds.attributes)
            #print(mRNA.attributes)
            #print(gene.attributes)

            processCodingGene(geneId, cds, mRNA)
            if( mRNA.id in codingGenes ):
                print("Duplicate entries for %s" % (mRNA.id,))
                duplicateCDSGenes += 1

            codingGenesInArea += 1

        else:
            assert(False)

    includedGenes += codingGenesInArea
    print("new, prev: ", codingGenesInArea, len(codingGenes)-prev)
    prev = len(codingGenes)
    
    print(area, codingGenesInArea)
    #print(list(codingGenes)[1:10])

print("%d genes included, %d skipped" % (includedGenes, skippedGenes))
if duplicateCDSGenes:
    print("Warning: found %d multiple-CDS features (more than one CDS under the same gene)" % duplicateCDSGenes )

#print(len(codingGenes))

# Write file containing selected gene-ids:
if args.output_gene_ids:
    with open(args.output_gene_ids, "w") as out:
        for geneId in sorted(codingGenes):
            out.write("%s\n" % geneId)
