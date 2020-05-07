from csv import reader
import codecs
import requests
from time import sleep
import xml.etree.ElementTree as ET
from data_helpers import SpeciesCDSSource, CDSHelper
from genome_model import getGenomeModelFromCache

#from data_helpers import CDSHelper


taxId = 511145
fns = (('./data/WebGesTer/ecoli/L_shaped_greatestdGcmp.fixed', '+'), ('./data/WebGesTer/ecoli/L_shaped_greatestdGreg.fixed', '-'))
requestDelaySeconds = 1.9

def fetchGenpeptRecord( genProtAccession ):
    requestUri = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={}&rettype=gp&retmode=xml".format( genProtAccession )

    sleep(requestDelaySeconds)
    r = requests.get( requestUri )
    if( r.status_code != 200 ):
        raise Exception("Entrez returned HTTP code %d, content %s" % r.status_code, r.text)
    
    return r.text # codecs.decode(r.text, 'utf-8', 'ignore')


def getLocusTag( genProtAccession ):
    xml = fetchGenpeptRecord( genProtAccession )

    root = None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        fn = 'convert_webgester_list.py.fetchGenPept_document.xml'
        f = open(fn, 'w')
        f.write(xml)
        f.close()
        print("XML error in report; saved to %s" % fn)
        raise e

    elems = root.findall(".//GBSeq/GBSeq_feature-table/GBFeature/GBFeature_quals/GBQualifier[GBQualifier_name=\"locus_tag\"]/GBQualifier_value")
    
    if elems:
        return elems[0].text
    else:
        return None


altIdentifiers = {}
def getIdentifiersConversionTableUsingGff3():
    global altIdentifiers

    if altIdentifiers:
        return altIdentifiers

    gm = getGenomeModelFromCache( taxId )
    
    for protId in SpeciesCDSSource(taxId):
        cds = CDSHelper( taxId, protId )
        geneId = cds.getGeneId()
        alts = gm.findEquivalentIdentifiers( geneId )
        for i in alts:
            altIdentifiers[i] = protId
        altIdentifiers[geneId] = protId
        
def convertAlternateId(identifier):
    return getIdentifiersConversionTableUsingGff3().get(identifier, None)
    
##613]     /No=822 /LP=3248408 /US=ggccggag /B=cca /DS=uuccggcc /T=uuaucccuca /USL=8 /DSL=8 /SL=8 /BL=3 /Mm=0 /Gp=0 /DG=-12.31 /G=yqjK /G>=3248099 /G<=3248398 /DS>=10 /DS<=29 /DM=19.5 /D=product=conserved protein; protein_id=NP_417571.1; GI=16130995
##123]     /No=170 /LP=696725 /US=acagccccgg /B=ugaaau /DS=ccggggcugu /T=uucaguuauu /USL=10 /DSL=10 /SL=10 /BL=6 /Mm=0 /Gp=0 /DG=-22.47 /G=asnB /G>=698400 /G<=696736 /DS>=11 /DS<=36 /DM=23.5 /D=product=asparagine synthetase B; protein_id=NP_415200.1; GI=16128650product=asparagine synthetase B; 
##777]     /No=1041 /LP=4135265 /US=cuggu(gg)ggc(g)uguu /B=uuaucau /DS=ggcggcu(aa)accag /T=guuuccagau /USL=15 /DSL=14 /SL=12 /BL=7 /Mm=2 /Gp=1 /DG=-13.00 /G=yijE /G>=4134131 /G<=4135036 /DS>=229 /DS<=265 /DM=247 /D=product=inner membrane protein, predicted permease; protein_id=NP_418378.4; GI=90111667


out = []
def processProteinRecord( rec, strand ):

    altIds = getIdentifiersConversionTableUsingGff3()
    
    atts = rec["D"]
    
    if "protein_id" not in atts: # Ignore non protein-coding genes
        return
    
    GI         = atts["GI"]
    assert(int(GI))              # make sure GI is convertible to int (since the source files had corruption problems I had to fix manually)
    proteinAcc = atts["protein_id"]
    product    = atts["product"]

    locus = getLocusTag( proteinAcc )

    x = convertAlternateId( locus )
    
    rec = ( x, GI, proteinAcc, locus, strand, product )
    print("\t".join(map(str, rec)))

    #cds = CDSHelper( taxid, proteinId )
    #print( "--> {} <--".format( cds.length()) )
    out.append( rec )
    

for fn, strand in fns:
    # Parse the unfortunate format used by WebGesTer-DB
    
    with open(fn) as csvfile:
        lineNo = 0
        #---------------------------------------------------------------
        # Parse level 1 (CSV, separeted by tabs)
        for row in reader( csvfile, delimiter='\t' ):
            lineNo += 1
            if lineNo==1: continue

            #---------------------------------------------------------------
            # Parse level 2 (key=value pairs, separated by '/')
            # note: some text fields contain internal '/' characters...
            pairs = [x.strip() for x in row[1].split(' /') if x]
            converted = {}
            for pair in pairs:
                att, val = pair.split('=', maxsplit=1)  # split on first '='
                converted[att] = val

            #---------------------------------------------------------------
            # Parse level 3
            if 'D' in converted:
                parts = converted['D'].split('; ')
                Dfields = {}
                for part in parts:
                    if part.find('=') != -1:
                        att, val = part.split('=', maxsplit=1)
                        Dfields[att] = val
                converted['D'] = Dfields

            processProteinRecord( converted, strand )
                
            

            

