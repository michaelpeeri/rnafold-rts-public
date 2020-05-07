from builtins import str
import os
import sys
import re
import subprocess
import codecs
import json
from datetime import datetime
from getpass import getuser
from ensembl_ftp import EnsemblFTP
from data_helpers import r, getSpeciesName, speciesNameKey, speciesTaxIdKey, speciesTranslationTableKey, speciesCDSList, setSpeciesProperty
import mysql_rnafold as db
from config import ensembl_data_dir
from genome_model import GenomeModel, CDSWith3PrimeSequencesWithDownstreamFeatureSource, isValidCodingSequence
from store_seqs_from_fasta import storeSeqInDB



# ~/anaconda2/bin/python2 gff3_extract_coding_genes.py --gff ./data/Ensembl/Twhipplei/Tropheryma_whipplei_str_twist.ASM748v1.36.gff3.gz --variant Ensembl --output-gene-ids ./data/Ensembl/coding_genes_list.Twhi.list --transl-table 11
def processGff3(gff3Filename, genesListFilename, args):
    print("Processing gff3 file...")
    output = []
    try:
        output = subprocess.check_output((sys.executable, "gff3_extract_coding_genes.py",
                                          "--gff3", gff3Filename,
                                          "--variant", args.variant, #"Ensembl",
                                          "--output-gene-ids", genesListFilename,
                                          "--transl-table", str(args.nuclear_genetic_code) ), shell=False)
    except subprocess.CalledProcessError as e:
        print(e)
        print("--" * 20)
        print("Process output: ")
        print("--" * 20)
        for line in output:
            print(line)
        print("--" * 20)
        print(e)

def loadCDSwith3UTRSequences(genomeModel, args, dryRun:bool =True) -> int:

    count = 0
    skippedCount = 0
    for (feat,region,cdsSeq,utr3Seq,gap,nextCdsSeq,nextCdsStrand) in CDSWith3PrimeSequencesWithDownstreamFeatureSource( genomeModel, debug=False ):

        #Interval(2598882, 2599761, {'strand': '-', 'props': '{"ID":["CDS:AAC75531"],"Parent":["transcript:AAC75531"],"protein_id":["AAC75531"]}'})
        props = json.loads( feat.data['props'] )
        proteinId = props['protein_id'][0]
        assert(len(proteinId)>3)
        print(proteinId)

        if not isValidCodingSequence( cdsSeq, geneticCode=genomeModel.getGeneticCode() ):
            skippedCount += 1
            continue
        #assert( isValidCodingSequence( cdsSeq, geneticCode=genomeModel.getGeneticCode() ))
        if not isValidCodingSequence( nextCdsSeq, geneticCode=genomeModel.getGeneticCode() ):
            skippedCount += 1
            continue
        #assert( isValidCodingSequence( nextCdsSeq, geneticCode=genomeModel.getGeneticCode() ))
        #cdsAASeq     = cdsSeq.translate(    table=genomeModel.getGeneticCode())
        #nextCdsAASeq = nextCdsSeq.translate(table=genomeModel.getGeneticCode())
        #assert(str(cdsAASeq)[:-1].find('*')==-1)
        #assert(str(nextCdsAASeq)[:-1].find('*')==-1)

        assert( feat.data['strand'] in ('+', '-') )
        assert( nextCdsStrand in ('+', '-') )
        nextCDSonOppositeStrand = feat.data['strand'] != nextCdsStrand
        if nextCDSonOppositeStrand:
            nextCdsSeq = nextCdsSeq.reverse_complement()

        if gap>=0:
            ntSeq = str(cdsSeq) + str(utr3Seq) + str(nextCdsSeq)
        else:
            assert(len(utr3Seq)==0)
            ntSeq = str(cdsSeq) + str(nextCdsSeq)[-gap:]
            
        count += 1

        cdsBoundaries = (region['curr-feature-start'],region['curr-feature-end'],region['downstream-feature-end'],region['downstream-feature-start'])
        if dryRun == False:
            storeSeqInDB(nucSeq = ntSeq,
                         taxId = args.taxid,
                         proteinId = proteinId,
                         nextCDSonOppositeStrand = nextCDSonOppositeStrand,
                         cdsLengthNt = len(cdsSeq),
                         flankingRegionLengthNt = gap,
                         genomeCoords = (min(cdsBoundaries), max(cdsBoundaries)),
                         seqSourceTag = db.Sources.CDSwith3primeFlankingRegion_DontExcludeNextORF)

    print("{} Ok, {} skipped".format(count, skippedCount))
    return count

        

    
def ingestGenome(taxid, args):

    if not args.dont_add_annotations:
        # Sanity test 1 -- this species doesn't already exist in the DB
        if r.exists(speciesNameKey % args.taxid):
            if r.exists(speciesCDSList % args.taxid):
                raise Exception("Species with taxid=%d already exists as '%s'" % (args.taxid, getSpeciesName(args.taxid)))
            else:
                pass # Name defined but no CDS list; maybe an earlier run of this script was interrupted?

    # Sanity test 2 -- genes list file doesn't already exists (if it does, short name may have been reused...)
    #genesListFilename = "%s/coding_genes_list.%s.list" % (ensembl_data_dir, args.short_name)
    #print(genesListFilename)
    if args.fetch_ftp_files:
        ftp = EnsemblFTP(args.local_name, args.remote_name, release=args.release, section=args.section, subsection=args.subsection, server=args.server)
        #assert(not os.path.exists(ftp.getLocalDir()))

    # Sanity tests passed

    genomefn = None
    cdsfn    = None
    cdnafn   = None
    gff3fn   = None
    
    # Step 1 - get files from Ensembl FTP
    if( args.variant == "Ensembl" and args.fetch_ftp_files ):
    
        (genomefn, cdsfn, cdnafn, gff3fn) = ftp.fetchAll()
        ftp.close()

        assert( os.path.exists(genomefn) and os.path.isfile(genomefn) )
        assert( os.path.exists(cdsfn)    and os.path.isfile(cdsfn) )
        assert( os.path.exists(cdnafn)   and os.path.isfile(cdnafn) )
        assert( os.path.exists(gff3fn)   and os.path.isfile(gff3fn) )
    else:
        gff3fn   = args.gff3
        genomefn = args.genome

    # Step 2 - parse GFF3 file to yield list of acceptable CDS genes
    #
    #processGff3(gff3fn, genesListFilename, args)

    #numGenesReturnedFromGff3 = None
    #with open(genesListFilename, "r") as f:
    #    numGenesReturnedFromGff3 = len(f.readlines())
    #if numGenesReturnedFromGff3 < 400:
    #    raise Exception("Processing gff3 file only yielded %d results; aborting" % numGenesReturnedFromGff3)
    #print("%d protein-coding genes found in gff3" % numGenesReturnedFromGff3)

    
    # Step 3 - Add required annotations for this species to redis DB
    if not args.dont_add_annotations:
        # TODO - add redis items here
        #redis-cli -h power5 -a rnafold set "species:taxid:203267:name" "Tropheryma whipplei str. Twist"
        r.set(speciesNameKey % args.taxid, args.full_name)

        #redis-cli -h power5 -a rnafold set "species:name:Tropheryma whipplei str. Twist:taxid" "203267"
        r.set(speciesTaxIdKey % args.full_name, args.taxid)

        #redis-cli -h power5 -a rnafold set "species:taxid:203267:genomic-transl-table"  "11"
        r.set(speciesTranslationTableKey % args.taxid, args.nuclear_genetic_code)

        addSupportingAnnotationsForGenome(args)
    

    gm = GenomeModel(
        sequenceFile = genomefn,
        gffFile=gff3fn,
        isLinear=False,
        variant=args.variant,
        geneticCode=args.nuclear_genetic_code )

    print( "Found {} genes".format( gm.numGenes() ))

    if args.dont_load_sequences:  return 0

    # Step 4 - Load CDS sequences to DB
    print("Doing trial run for gene loading...")
    cdsLoadedCount = loadCDSwith3UTRSequences(gm, args, dryRun=True)
    if cdsLoadedCount > 400:
        print("Trial run succeeded.")
        print("Performing actual gene loading...")
        cdsLoadedCount = loadCDSwith3UTRSequences(gm, args, dryRun=False)
        print("Loaded %d CDS genes..." % cdsLoadedCount)
        
    else:
        print("Dry-run loading may have encountered errors; aborting without actual load...")
        return -1

    return 0

def addSupportingAnnotationsForGenome( args, overwrite=False ):
    #def setSpeciesProperty(taxId, propName, propVal, source, overwrite=True):

    propSource = "Manual entry; {}; {}".format( getuser(), datetime.now().isoformat(' ') )

    assert( os.path.exists( args.genome ))
    setSpeciesProperty( args.taxid,
                        "genome-seq-path",
                        os.path.abspath( args.genome ),
                        propSource,
                        overwrite=overwrite )
                        
    assert( os.path.exists( args.gff3 ))
    setSpeciesProperty( args.taxid,
                        "genome-annot-path",
                        os.path.abspath( args.gff3 ),
                        propSource,
                        overwrite=overwrite )
    
    setSpeciesProperty( args.taxid,
                        "genome-annot-variant",
                        args.variant,
                        propSource,
                        overwrite=overwrite )

def ingestMultipleGenome(args):

    failCount = 0
    successCount = 0

    allTaxids = (args.taxid,)  # processing multiple genomes is not supported yet...
    
    for taxid in allTaxids:
        #try:
        ingestGenome(taxid, args)
        successCount += 1
            
        #except Exception as e:
        #    print(e)
        #    failCount += 1

    if not failCount:
        print("Processed {} genomes.".format(successCount))
        return 0
    
    else:
        print("{} genomes failed, {} genomes succeeded.".format(failCount, successCount))
        return -1

    
        
            

_knownBoolVals = {"true":True, "false":False}
def parseBool(val):
    _val = val.lower()
    if _val in _knownBoolVals:
        return _knownBoolVals[_val]
    else:
        raise Exception("Unknown bool value '%s'" % _val)
    

def standaloneRun():
    import argparse
    import sys
    
    argsParser = argparse.ArgumentParser()
    #argsParser.add_argument("--verbose", type=int, default=0)
    argsParser.add_argument("--local-name", type=str, required=False)
    argsParser.add_argument("--remote-name", type=str, required=False)
    argsParser.add_argument("--release", type=int, default=95)
    argsParser.add_argument("--section", type=str, default="bacteria")
    argsParser.add_argument("--subsection", type=str, required=False)
    argsParser.add_argument("--nuclear-genetic-code", type=int, required=True)
    argsParser.add_argument("--taxid", type=int, required=True)
    argsParser.add_argument("--full-name", type=str, required=True)
    argsParser.add_argument("--short-name", type=str, required=True)
    argsParser.add_argument("--variant", type=str, default="Ensembl")
    argsParser.add_argument("--fetch-ftp-files", type=parseBool, default=True)
    argsParser.add_argument("--gff3", type=str, required=False)
    argsParser.add_argument("--genome", type=str, required=False)
    argsParser.add_argument("--cds-with-3utr", action="store_true", default=False)
    argsParser.add_argument("--ignore-id-check", action="store_true", default=False)
    argsParser.add_argument("--server", type=str, required=False, default=None)
    argsParser.add_argument("--add-annotations-only", action="store_true", default=False) 
    argsParser.add_argument("--dont-load-sequences", action="store_true", default=False)
    argsParser.add_argument("--dont-add-annotations", action="store_true", default=False)
   
    args = argsParser.parse_args()

    print(args)

    print(args.fetch_ftp_files)

    if( args.add_annotations_only ):
        sys.exit( addSupportingAnnotationsForGenome(args) )

    if( args.fetch_ftp_files ):
        if( (not args.local_name) or (not args.remote_name) ):
            raise Exception("--remote-name and --local-name are required when --fetch-ftp-files=True")
        
        if( args.gff3 or args.genome ):
            raise Exception("--gff3 and --genome are not supported when --fetch-ftp-files=True")
        
        if args.variant != "Ensembl":
            raise Exception("Only --variant=Ensembl is supported with --fetch-ftp-files=True")
    else:
        if( args.local_name or args.remote_name ):
            raise Exception("--remote-name and --local-name cannot be used when --fetch-ftp-files=False")
        
        if( (not args.gff3) or (not args.genome) ):
            raise Exception("--gff3 and --genome are required when --fetch-ftp-files=False")
        

    sys.exit(ingestMultipleGenome(args))
    
    
if __name__=="__main__":
    standaloneRun()
