# Renamed from test_gff_extract_coding_genes.py
from __future__ import print_function
from builtins import map
import argparse
from collections import Counter
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
argsParser.add_argument("--gff3")
argsParser.add_argument("--variant", type=parseOption(set(("yeastgenome", "NCBI", "Ensembl", "JGI")), "variant"))
argsParser.add_argument("--output-gene-ids")
argsParser.add_argument("--areas", type=parseList(int))
argsParser.add_argument("--transl-table", type=int)
argsParser.add_argument("--alt-protein-ids", type=parseOption(set(("locus_tag",)), "alt-protein-id"))
args = argsParser.parse_args()
assert(args.transl_table)

print(args)

print("Loading gff3 file...")
db = gff.createGffDb(args.gff3, args.variant)
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
        
    elif( args.variant=="JGI"):
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

    elif( args.variant=="JGI"):
        for cds in db.features_of_type('CDS'):
            #print("CDS.id: %s" % cds.id)
            mRNA = None
            gene = None
            for p in db.parents(cds.id):
                #print("p.featuretype: %s" % (p.featuretype))

                #assert(p.featuretype=="mRNA")
                if p.featuretype=="mRNA":
                    mRNA = p
                elif p.featuretype=="gene":
                    gene = p
                else:
                    print("p.featuretype: %s" % (p.featuretype))

                #for q in db.parents(p.id):
                #    print("q.featuretype: %s" % (q.featuretype))
                #    assert(q.featuretype=="gene")
                #    gene = q


            if cds and mRNA and gene:
                yield cds, mRNA, gene
                # Note: will yield the same gene multiple time, once for every exon (CDS feature). This will be handled downstream
            else:
                print("Unhandled:\ncds=%s\nmRNA=%s\ngene=%s\n" % (cds, mRNA, gene))
                print(len(list(db.parents(cds.id))))
                assert(False)
                
                #if p.featuretype=='gene':
                #    gene = p

    else:
        assert(False)


skipReasons = Counter()
        
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

            # cds: KV784353.1 Genbank CDS     64126   64258   .       +       0       ID=cds19;Dbxref=InterPro:IPR002110,JGIDB:Fracy1_231823,NCBI_GP:OEU21681.1;Name=OEU21681.1;gbkey=CDS;Parent=rna19;product=ankyrin;protein_id=OEU21681.1
            # mRNA: KV784353.1        Genbank mRNA    61430   65507   .       +       .       ID=rna19;gbkey=mRNA;end_range=.,65507;partial=true;start_range=.,61430;Parent=gene19;product=ankyrin
            # gene: KV784353.1        Genbank gene    61430   65507   .       +       .       ID=gene19;Name=FRACYDRAFT_231823;gbkey=Gene;end_range=.,65507;gene_biotype=protein_coding;locus_tag=FRACYDRAFT_231823;partial=true;start_range=.,61430

            skipReasons.update(("total",))

            print("--" * 20)
            print("cds: %s" % cds)
            print("mRNA: %s" % mRNA)
            print("gene: %s" % gene)
            
            # TODO - FIX THIS!!!!
            #if(mRNA.featuretype != "gene"):
            #    skippedGenes += 1
            #    continue

            #assert(mRNA.featuretype=="gene")

            #print(mRNA)

            # Skip pseudo-genes
            if( gene.attributes['gene_biotype'][0] == 'pseudogene' ):
                skipReasons.update(("gene_pseudogene",))
                skippedGenes += 1
                continue
            
            if( gene.attributes['gene_biotype'][0] != 'protein_coding' ):
                skipReasons.update(("gene_biotype",))
                skippedGenes += 1
                continue

            if( 'pseudo' in cds.attributes and cds.attributes['pseudo'][0] == 'true' ):
                skipReasons.update(("cds_pseudo",))
                skippedGenes +=1
                continue

            if( 'pseudo' in gene.attributes and gene.attributes['pseudo'][0] == 'true' ):
                skipReasons.update(("gene_pseudo",))
                skippedGenes +=1
                continue

            if( 'partial' in mRNA.attributes and mRNA.attributes['partial'][0] == 'true' ):
                skipReasons.update(("mRNA_partial",))
                skippedGenes +=1
                continue
            
            if( 'transl_table' in cds.attributes and int(cds.attributes['transl_table'][0]) != args.transl_table ):
                skipReasons.update(("cds_transl_table",))
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
                skipReasons.update(("_Dbxrefs",))
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
            biotype = None

            if mRNA is None:
                skippedGenes += 1
                continue
            
            biotype = mRNA.attributes['biotype'][0]
            if( biotype != 'protein_coding' ):
                skippedGenes += 1
                continue

            geneId = mRNA.attributes['transcript_id'][0]
            if( not geneId ):
                print("Missing gene-id!")
                assert(False)

            processCodingGene(geneId, cds, mRNA)
            if( mRNA.id in codingGenes ):
                print("Duplicate entries for %s" % (mRNA.id,))
                duplicateCDSGenes += 1

            codingGenesInArea += 1

        elif(args.variant=="JGI"):

            geneId = mRNA.attributes['Name'][0]
            assert(len(geneId))

            if( geneId in codingGenes):
                continue
            
            processCodingGene(geneId, cds, mRNA)
            
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

print(skipReasons)
    
#print(len(codingGenes))

# Write file containing selected gene-ids:
if args.output_gene_ids:
    with open(args.output_gene_ids, "w") as out:
        for geneId in sorted(codingGenes):
            out.write("%s\n" % geneId)
