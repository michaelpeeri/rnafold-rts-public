# Import native sequences into DB, creating the related metadata in redis.
#
# Can also be used from command-line to import seqs directly from fasta:
# Command-line args: <taxId> <fastaFile> <type_cds|type_shuffle>
# Read a fasta file containing the sequences for a given species;
# Store the sequences in MySql, and add the sequence-ids to the metadata in redis.
#
from __future__ import print_function
from builtins import str
import sys
import re
import argparse
import gzip
import redis
import codecs
from Bio import SeqIO
from Bio.Alphabet import generic_dna
import config
import mysql_rnafold as db
from sqlalchemy import sql
from sqlalchemy.sql.expression import func
import nucleic_compress
import data_helpers

def parseOption(possibleValues, name):
    def checkOption(value):
        if value in possibleValues:
            return value
        else:
            raise argparse.ArgumentTypeError("Unknown %s '%s', allowed values: %s" % (name, value, ",".join(possibleValues)))
    return checkOption



# configuration
# redis keys
# TODO - unify with list from data_helpers.py
cdsSeqWith3UTRIdKey   = "CDS:taxid:%d:protid:%s:cds-3utr-seq-id"
seqLengthKey          = "CDS:taxid:%d:protid:%s:length-nt"
proteinIdKey          = "CDS:taxid:%d:protid:%s:protein-id"  # Used to store the protein-id for species in which a different main identifier is used
shuffledSeqIdsKey     = "CDS:taxid:%d:protid:%s:shuffled-seq-ids-v2"
cdsSeqChecksumKey     = "CDS:taxid:%d:protid:%s:cds-seq-checksum"
cdsOnlyLengthKey      = "CDS:taxid:%d:protid:%s:cds-length-nt"
flanking3utrLengthKey = "CDS:taxid:%d:protid:%s:3utr-flank-length-nt"
nextCdsOnOppositeStrandKey = "CDS:taxid:%d:protid:%s:next-cds-opp-strand"
#genomicCoordStartKey  = "CDS:taxid:%d:protid:%s:genomic-start"
#genomicCoordEndKey    = "CDS:taxid:%d:protid:%s:genomic-end"
#partialCDSKey        = "CDS:taxid:%d:protid:%s:partial"
speciesCDSList        = "species:taxid:%d:CDS"
regexLocusId = re.compile("([^.]+[.][^.]+)[.].*")


r = redis.StrictRedis(host=config.host, port=config.port, db=config.db, password=config.password)
# sequences server (mysql)
session = db.Session()


translateAmbiguousNucleotides        = str.maketrans("RrYyKkMmSsWwBbDdHhVv",     "nnnnnnnnnnnnnnnnnnnn")

def storeSeqInDB(nucSeq, taxId:int, proteinId:str, seqSourceTag:int, stopCodonPos:int=-1, genomeCoords:tuple=(), nextCDSonOppositeStrand:bool=None, cdsLengthNt:int=None, flankingRegionLengthNt:int=None ) -> None:

    # Compress the CDS sequence
    encodedCds = nucleic_compress.encode( str(nucSeq).translate( translateAmbiguousNucleotides ))  # convert all ambiguity codes to 'n' before compression
    # throws KeyError if encountering non-nucleic (acgtn) symbols 


    # Store the shuffled CDS sequence
    s1 = db.Sequence2(sequence=encodedCds, alphabet=db.Alphabets.RNA_Huff, source=seqSourceTag)
    session.add(s1)
    session.commit()

    newSequenceId = s1.id

    if( seqSourceTag == db.Sources.CDSwith3primeFlankingRegion_DontExcludeNextORF ):
        #
        # Add Native CDS with 3' flanking region
        #
        # Store the sequence id
        r.set(cdsSeqWith3UTRIdKey % (taxId, proteinId), newSequenceId )
        r.sadd(speciesCDSList % (taxId,), proteinId)

        # Store the full sequence length (e.g., CDS+flanking region+nextCDS)
        r.set(seqLengthKey % (taxId, proteinId), len(nucSeq) )

        # Store the canonical protein-id (in species where a different unique id is used)
        #if args.alt_protein_ids and altProteinId:
        #    r.set( proteinIdKey % (taxId, proteinId), altProteinId )
        
        # Store the CDS checksum
        crc1 = data_helpers.getCrc(nucSeq)
        r.set(cdsSeqChecksumKey % (taxId, proteinId), crc1)

        # Length of the "main" CDS only
        if not cdsLengthNt is None:
            r.set(cdsOnlyLengthKey % (taxId, proteinId), cdsLengthNt )

        # Length of the "gap" or "flanking region" in the 3'UTR
        if not flankingRegionLengthNt is None:
            r.set(flanking3utrLengthKey % (taxId, proteinId), flankingRegionLengthNt )

        if not nextCDSonOppositeStrand is None:
            r.set(nextCdsOnOppositeStrandKey % (taxId, proteinId), "1" if nextCDSonOppositeStrand else "0")
            
        #if genomeCoords:
        #    r.set(genomicCoordStartKey % (taxId, proteinId), genomeCoords[0])
        #    r.set(genomicCoordEndKey   % (taxId, proteinId), genomeCoords[1])
                        
    else:
        assert(False)


def standalone():
    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--taxid", type=int)
    argsParser.add_argument("--input")
    argsParser.add_argument("--variant", type=parseOption(set(("yeastgenome", "NCBI", "Ensembl", "JGI")), "variant"))
    argsParser.add_argument("--type", type=parseOption(set(("cds", "shuffle", "fixCDSkey")), "sequence type"))
    argsParser.add_argument("--dry-run", action="store_true", default=False)
    argsParser.add_argument("--output-fasta")
    argsParser.add_argument("--gene-ids-file")
    argsParser.add_argument("--alt-protein-ids", type=parseOption(set(("locus_tag",)), "alt-protein-id"))
    argsParser.add_argument("--headers-from-another-fasta")
    argsParser.add_argument("--ignore-id-check", action="store_true", default=False)
    args = argsParser.parse_args()

    if( args.output_fasta ):
        if( args.output_fasta == args.input ):
            raise Exception("Fasta output file cannot match input file!")

    #if( len(sys.argv) < 5 ):
    #    print("Usage: %s <taxid> <fasta-file> <fasta-variant> <cds|shuffle>" % (sys.argv[0],))
    #    sys.exit(-1)

    # command-line arguments
    taxId = args.taxid
    f = None
    if( args.input[-3:]==".gz"):
        f = gzip.open(args.input, "r")
    elif( args.input[-4:]==".bz2"):
        # TODO: impl this...
        assert(False)
    else:
        f = open(args.input, 'r')
    #sequenceFormat = args.variant
    sequenceType = args.type


    if( sequenceType=="cds" ):
        seqSourceTag = db.Sources.External
    elif( sequenceType=="shuffle" ):
        seqSourceTag = db.Sources.ShuffleCDSv2_matlab
    elif( sequenceType=="fixCDSkey" ):
        seqSourceTag = None
    else:
        raise Exception("Unknown sequence type '%s'"%sequenceType)


    # establish connections
    # metadata server (redis)
    #r = redis.StrictRedis(host=config.host, port=config.port, db=config.db, password=config.password)
    # sequences server (mysql)
    #session = db.Session()


    visitedProteinIds = set()

    assert(r.exists("species:taxid:%d:name" % taxId))

    if( seqSourceTag == db.Sources.External ):
        # Clear any previously imported CDSs...
        #r.delete(speciesCDSList % (taxId,))
        count = data_helpers.countSpeciesCDS(taxId)
        if( count > 0 and (not args.dry_run)):
            print("%d sequences already exist for specied %d. Aborting..." % (count, taxId))
            sys.exit(-1)
    elif( sequenceType=="fixCDSkey" ):
        r.delete(speciesCDSList % (taxId,))
        # Delete and reconstruct the CDS key
    else:
        assert( data_helpers.countSpeciesCDS(taxId) > 0 )



    reNuclearYeastGene = re.compile("Y[A-P][RL]\d+[CW](-[A-Z])?")
    geneIdsToInclude = set()
    if(args.gene_ids_file):
        with open(args.gene_ids_file, "r") as genesFile:
            for geneId in genesFile:
                geneIdsToInclude.add(geneId.rstrip())


    reNCBIattributes = re.compile("\[(\S+)=([^\]]+)\]")
    reNCBIbareheader = re.compile("\w+\|\w+\.\d+_cds_(\w+.\d+)_\d+")
    outRecords = []

    headersFromAnotherFasta = {}
    if args.headers_from_another_fasta:
        with open(args.headers_from_another_fasta, "r") as f2:
            for record in SeqIO.parse(f2, "fasta", alphabet=generic_dna):
                assert(not record.id in headersFromAnotherFasta)
                headersFromAnotherFasta[record.id] = record.description

    cdsCount = 0
    notFoundCount = 0
    skippedCount = 0
    validNucleotideChars = str.maketrans( "ACGTacgt", "%%%%%%%%" )
    #print("Opening fasta file: {}".format(f))
    for record in SeqIO.parse(f, "fasta", alphabet=generic_dna):
        #proteinId = regexLocusId.match(record.id).group(1) # Work-around for multiple-transcript identifiers in JGI's Chlamy genome

        if args.headers_from_another_fasta:
            record.description = headersFromAnotherFasta[record.id]

        numNonNucleotideChars = len(record.seq) - str(record.seq).translate( validNucleotideChars ).count("%")
        if numNonNucleotideChars:
            print("Skipping record %s, containing non-nucleotide or ambiguous symbols '%s'" % (record.id, numNonNucleotideChars))
            skippedCount +=1
            continue

        # yeastgenome.org - skip suspected pseudo-genes
        if(args.variant=="yeastgenome" and record.description.find("Dubious ORF") != -1):
            skippedCount += 1
            continue

        # yeastgenome.org - skip mitochondrial genes
        if(args.variant=="yeastgenome"):
            geneType = record.id[0]
            if geneType == "Q" or geneType == "R":
                skippedCount += 1
                continue

        # yeastgenome.org - verify gene-id conforms to: http://www.yeastgenome.org/help/community/nomenclature-conventions
        if(args.variant=="yeastgenome"):
            geneId = record.id
            assert(reNuclearYeastGene.match(geneId))

        # Obtain attributes mapping
        attributes = []
        if(args.variant=="NCBI"):
            attributes = dict(re.findall(reNCBIattributes, record.description))

        if(args.variant=="NCBI"):
            if( 'pseudo' in attributes and attributes['pseudo']=='true' ):
                print("Skipping pseudo-gene entry %s" % (record.id,))
                skippedCount += 1
                continue

        # Determine gene id
        proteinId = None
        additionalProteinIds = set()
        altProteinId = None
        if(args.variant=="yeastgenome"):
            proteinId = record.id
        elif(args.variant=="NCBI"):
            if( sequenceType=="shuffle" and not attributes):
                #Workaround for shuffle-seq files missing the header...
                #Extract the protein-id from sequence-id like this:
                #>lcl|NC_002516.2_cds_NP_064721.1_1
                if not args.alt_protein_ids:
                    proteinId = reNCBIbareheader.match(record.id).group(1)

                elif args.alt_protein_ids=="locus_tag":
                    if( 'locus_tag' not in attributes ):
                        print("Skipping entry %s missing locus_tag - %s" % (record.id, attributes))
                        skippedCount += 1
                        continue
                    proteinId = attributes['locus_tag']
                    print(proteinId)
                else:
                    assert False

            else:
                # Note - not currently used
                #if 'db_xref' in attributes:
                #    _db_xrefs = attributes['db_xref'].split(",")
                #    db_xrefs = dict(map( lambda x: tuple(x.split(":")), _db_xrefs))
                if not args.alt_protein_ids:
                    if( 'protein_id' not in attributes ):
                        print("Skipping entry %s missing protein_id - %s" % (record.id, attributes))
                        skippedCount += 1
                        continue

                    proteinId = attributes['protein_id']
                elif args.alt_protein_ids=="locus_tag":
                    if( 'locus_tag' not in attributes ):
                        print("Skipping entry %s missing locus_tag - %s" % (record.id, attributes))
                        skippedCount += 1
                        continue
                    proteinId = attributes['locus_tag']

                    if( 'protein_id' in attributes ):
                        altProteinId = attributes['protein_id']

                else:
                    assert(False)

        elif(args.variant=="Ensembl"):
            # Sample id: ABD29211.1
            dotPos = record.id.rfind('.')
            if( dotPos > 3 ):
                proteinId = record.id[:dotPos]
                additionalProteinIds.add(record.id) # also allow matching the full format (including the transcript-id) - some CDS files include it...

            else:
                proteinId = record.id

        elif(args.variant=="JGI"):
            # Variant 1 (Phytozome, Mpus)
            #  (gff3):  60050
            #  (fasta): 60050
            # Variant 2 (Phytozome, Dsal)
            #  (gff3):  Dusal.1637s00001.1
            #  (fasta): Dusal.1637s00001.1
            # Variant 3:
            #  (gff3):  jgi.p|Ostta1115_2|10314
            #  (fasta): jgi|Ostta1115_2|10314|CE10313_131

            proteinId = record.id

            if record.id.startswith("jgi|"):
                parts = record.id.split('|')
                parts[0] = 'jgi.p'                                 # add the '.p'
                additionalProteinIds.add( '|'.join( parts[:3] ) )  # drop the suffix (parts[4])

        else:
            assert(False)

        if not args.ignore_id_check:
            assert(len(proteinId)>2)

        # Skip sequences that have non-standard translations
        if(args.variant=="NCBI"):
            if "transl_except" in attributes:
                print("Skipping %s (because of transl_except)" % (proteinId,))
                skippedCount += 1
                continue

        # If an inclusion list (white list) is defined, skip sequences missing from it
        if args.gene_ids_file:
            if( proteinId not in geneIdsToInclude):
                # Also try the additional ids
                if( not geneIdsToInclude.intersection( additionalProteinIds ) ):
                    print("Skipping %s (sequence %s, alternate ids=%s)" % (proteinId, record.id, list(additionalProteinIds)))
                    skippedCount += 1
                    continue

        print("Inserting %s (sequence %s)..." % (proteinId, record.id))

        # Verify there are no duplicates entries
        if(proteinId in visitedProteinIds):
            print("MULTIPLE Entry: %s", proteinId)
            skippedCount +=1 
            continue
        #assert(proteinId not in visitedProteinIds)
        visitedProteinIds.add(proteinId)

        # Write the filtered sequences into an output file (if needed)
        # Note - this also works in dry-run...
        if( args.output_fasta ):
            outRecords.append(record)

        if( args.dry_run ):
            continue

        if( sequenceType=="fixCDSkey" ):
            cds = data_helpers.CDSHelper(taxId, proteinId )
            seqId = cds.seqId()
            if(not seqId is None):
                r.sadd(speciesCDSList % (taxId,), proteinId)
            else:
                print("Couldn't find entry for proteinId=%s" % proteinId)

            continue # Skip the rest of the processing...


        storeSeqInDB( nucSeq=record.seq, taxId=taxId, proteinId=proteinId, seqSourceTag=seqSourceTag )

        cdsCount += 1

    if( notFoundCount + skippedCount > 0):
        print("Warning: %d entries skipped and %d entries not found" % (skippedCount, notFoundCount))

    print("Processed %d CDS entries" % (cdsCount,))
    print("(out of %d CDS entries for this species)" % (r.scard("species:taxid:%d:CDS" % (taxId,))))


    if( args.output_fasta ):
        with open(args.output_fasta, "w") as outfile:
            out = SeqIO.write( outRecords, outfile, "fasta")


                
    
if __name__=="__main__":
    import sys
    sys.exit(standalone())

