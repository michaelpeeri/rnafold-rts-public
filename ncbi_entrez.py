from __future__ import division
# Get stuff from NCBI Entrez
# Try to use biopython libs whenever possible...
from builtins import map
import codecs
import xml.etree.ElementTree as ET
#import subprocess
#import zlib
import re
from pyparsing import Word, Literal, delimitedList, printables, ParseException, Suppress, Group
from Bio import Entrez
import requests
from time import sleep
from ete3 import NCBITaxa
from data_helpers import getSpeciesName, getSpeciesTaxonomicGroup, allSpeciesSource, setSpeciesProperty


#------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------
Entrez.email = "mich1@post.tau.ac.il"
curlPath = "/home/michael/anaconda2/bin/curl"
requestDelaySeconds = 1

limitSpecies = frozenset()
#limitSpecies = frozenset((1330330,521045,391009))
#------------------------------------------------------------------

reInteger = re.compile("^-?\d+$")
reFloat   = re.compile("^-?\d+[.]\d+$")
reRange   = re.compile("^(\d+)-(\d+)C?$")

# html
# body
# div class="grid"
# div class="col twelve_col nomargin shadow"
# form id="EntrezForm"
# div
# div id="maincontent"
# div class="content"
# div
# div class="rprt"
# div class="MainBody"
# table
# tbody
# tr
# td
# div class="rprt-section" id="mmtest1"
# div class="ui-helper-reset"
# div class="rprt-section-body ui-widget ui-ncbitoggler-open ......"
# div class="summary"
# table class="summary"
# tbody
#
# ---------------------
# tr
# td
# b text "Morphology: "
#
# td
# text "........."
#
# ---------------------
# tr
# td
# b text "Environment: "
#
# td
# text "............"
# ---------------------
#
# tr
# td
# b test "Statistics: "
# td
# text "................."
#
# tr
# td colspan="99"
# text "...................."
#
# tr
# td colspan="99"
# text "....................."
#

elementsToRemove = ['<B>', '</B>', '<b>', '</b>', '<I>', '</I>', '<i>', '</i>', '&acc', '&chr']
"""
NCBI HTMLs aren't valid XML (though they are declared as such): at the moment (07-2017) they contain mismatched formatting elements (e.g., <b><i>some text</b></i>).
As a work-around, we strip some formatting (working on the flat string, i.e., before parsing).
"""
def stripHTMLformatting(html):
    for f in elementsToRemove:
        html = html.replace( f, ' ' )
    return html


missingImgCloseTag = (('<img alt="ftp-genbank" src="/sutils/static/ProtMap/mark5.gif" title="GenBank FTP">', '<img alt="ftp-genbank" src="/sutils/static/ProtMap/mark5.gif" title="GenBank FTP"/>'),
                      ('<img alt="ftp-refseq" src="/sutils/static/ProtMap/mark4.gif" title="RefSeq FTP">', '<img alt="ftp-refseq" src="/sutils/static/ProtMap/mark4.gif" title="RefSeq FTP"/>'))
def fixMissingImgCloseTags(html):
    for _from, _to in missingImgCloseTag:
        html = html.replace( _from, _to )
    return html

def collectInnerText(elem):
    return "".join(elem.itertext()).strip()

# Parse using pyparsing grammar
word=Word(printables+' '+'\t', excludeChars=',:')
keyValPair = Group(word + Suppress(Literal(':')) + word)
expr = delimitedList( keyValPair, delim=',' )
"""
Convert compound values (encoded as "key:value, key:value") to dict
"""
def parseCompoundVal(val):
    if val.find(':') == -1:  # simple strings stay strings
        return val

    #print("--- %s ---" % val)

    try:
        res = expr.parseString(val)
        val = dict(res.asList())
        if not val:
            val = ""
    except ParseException as e:
        #print("--------------------")
        #print(val)
        #print(e)
        pass

    if type(val)==type({}):
        converted = {}
        for k, v in list(val.items()):
            v = v.strip()
            if reInteger.match(v):
                v = int(v)
            elif reFloat.match(v):
                v = float(v)
            elif reRange.match(v):
                match = reRange.match(v)
                v = (int(match.group(1)), int(match.group(2)))
            converted[k] = v
        val = converted
        
    return val

def parseNCBIGenomeHTML_fetchSummaryReport(html):
    root = None
    try:
        root = ET.fromstring(stripHTMLformatting(html))
    except ET.ParseError as e:
        fn = 'parseNCBIGenomeHTML_fetchSummaryReport_document.xml'
        f = open(fn, 'w')
        f.write(html)
        f.close()
        print("XML error in report; saved to %s" % fn)
        raise e
        
    summaryTables = root.findall(".//{http://www.w3.org/1999/xhtml}div[@class='summary']/{http://www.w3.org/1999/xhtml}table[@class='summary']")

    props = {}

    for table in summaryTables:   # Some pages contain two tables (one title 'Summary' and another titled 'Genome Neighbors')
        key = ""
        val = ""
        # The table encodes keys and values
        for tr in table.findall(".//{http://www.w3.org/1999/xhtml}tr"):
            tds = tr.findall(".//{http://www.w3.org/1999/xhtml}td")
            
            if len(tds)==2:    # Common case: <tr> <td>Key:</td><td>Value</td> </tr>
                key = collectInnerText(tds[0])
                val = collectInnerText(tds[1])
                
            elif len(tds)==1:  # second case: <tr> <td span="99">Additional value (for previous key)</td> </tr>
                val = collectInnerText(tds[0])
                
            else:
                pass

            val = parseCompoundVal(val)  # convert compound values (encoded as "key:value, key:value") to dict

            if not key:
                pass
                #print("Ignoring empty key with val=%s" % val)
            else:
                if key in props:
                    #print("Will join existing val %s to new val %s (key %s)" % (props[key], val, key))
                    if type(props[key])==type(""):
                        props[key] = props[key] + ", " + val
                    elif type(props[key])==type({}):
                        props[key].update( val )
                    else:
                        assert(False)
                else:
                    props[key] = val

    return props



def wrapTableFragmentAsXML(html):
    pre = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<table>
"""
    post = """</table>
</html>
"""
    return pre + html + post


reAssemblyUriMatchGenomeAndAssembly = re.compile("/genome/(\d+)[?]genome_assembly_id=(\d+)")
"""
Parse genome assemblies table; returns (genome-id, assembly-id)
This uses several assumptions (documented inside).

To obtain the assemblies table (i.e., the input for this function), use something like this (obtain using Chrome's 'copy as curl'):

curl 'https://www.ncbi.nlm.nih.gov/genomes/Genome2BE/genome2srv.cgi?action=GetGenomes4Grid&genome_id=15&genome_assembly_id=&king=Eukaryota&mode=2&flags=1&page=1&pageSize=100' -H 'DNT: 1' -H 'Accept-Encoding: gzip, deflate, br' -H 'Accept-Language: en-US,en;q=0.8' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/59.0.3071.109 Chrome/59.0.3071.109 Safari/537.36' -H 'Accept: */*' -H 'Referer: https://www.ncbi.nlm.nih.gov/genome/genomes/15' -H 'X-Requested-With: XMLHttpRequest' -H 'Connection: keep-alive' --compressed
"""
def parseNCBIGenomeAssembliesHTML_fetchMainAssembly(html):
    root = ET.fromstring(html)

    trs = root.findall("./{http://www.w3.org/1999/xhtml}table/{http://www.w3.org/1999/xhtml}tr")
    if not trs:
        raise Exception("Failed to find any assemblie records in table")

    # Note: We are making two assumptions here:
    # 1) The first assembly in the list is the 'reference' assembly (i.e., the one you would get if you click "Reference genome" at the top of one of the other genome pages)
    # 2) The 'reference' assembly will contain all metadata available
    #
    selectedAssembly = 0 # Select the first assembly
    
    assemblyRec = trs[selectedAssembly]
    tds = assemblyRec.findall("./{http://www.w3.org/1999/xhtml}td")

    linkCell = tds[0]
    link = linkCell.findall("./{http://www.w3.org/1999/xhtml}a")[0]
    href = link.get("href")
    if not href:
        raise Exception("Failed to obtain link to reference assembly")

    match = reAssemblyUriMatchGenomeAndAssembly.match(href)
    if not match:
        raise Exception("Unrecognized structure of link to reference assembly")

    genomeId   = int(match.group(1))
    assemblyId = int(match.group(2))
        
    return (genomeId, assemblyId)


"""
Convert Entrez taxId to Entrez Genome-id (using the Entrez API wrapper)
"""
def taxIdToGenomeId(taxId):
    speciesName = getSpeciesName(taxId)
    ##speciesName = "Saccharomyces cerevisiae"
    ##speciesName = "Methanocaldococcus jannaschii DSM 2661"
    sleep(requestDelaySeconds)
    handle = Entrez.esearch(db="genome", retmax=10, term="\"%s\"[ORGN]" % speciesName)
    record = Entrez.read(handle)
    handle.close()

    return list(map(int, record['IdList']))


# """
# curl 'https://www.ncbi.nlm.nih.gov/genomes/Genome2BE/genome2srv.cgi?action=GetGenomes4Grid&genome_id=15&genome_assembly_id=&king=Eukaryota&mode=2&flags=1&page=1&pageSize=100' -H 'DNT: 1' -H 'Accept-Encoding: gzip, deflate, br' -H 'Accept-Language: en-US,en;q=0.8' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/59.0.3071.109 Chrome/59.0.3071.109 Safari/537.36' -H 'Accept: */*' -H 'Referer: https://www.ncbi.nlm.nih.gov/genome/genomes/15' -H 'X-Requested-With: XMLHttpRequest' -H 'Connection: keep-alive' --compressed
# """
# def fetchEntrezAssembliesTableForSpecies_curl(genomeId):
#     command = curlPath
#     args = ("""https://www.ncbi.nlm.nih.gov/genomes/Genome2BE/genome2srv.cgi?action=GetGenomes4Grid&genome_id=%d&genome_assembly_id=&mode=2&flags=1&page=1&pageSize=20""" % genomeId,
#             "-H",
#             """DNT: 1""",
#             "-H",
#             """Accept-Encoding: gzip, deflate, br""",
#             "-H",
#             """Accept-Language: en-US,en;q=0.8""",
#             "-H",
#             """User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/59.0.3071.109 Chrome/59.0.3071.109 Safari/537.36""",
#             "-H",
#             """Accept: */*""",
#             "-H",
#             """Referer: https://www.ncbi.nlm.nih.gov/genome/genomes/%d""" % genomeId,
#             "-H",
#             """X-Requested-With: XMLHttpRequest""",
#             "-H",
#             """Connection: keep-alive""",
#             "--compressed")
#     out = None
#     try:
#         out = subprocess.check_output( (command,)+args , stderr=subprocess.STDOUT, shell=False )
        
#         #out = subprocess.check_output( ('curl', '--help'), stderr=subprocess.STDOUT, shell=False )
#         #out = subprocess.check_output( command, shell=False)
#     except subprocess.CalledProcessError as e:
#         print(e)
#         print(e.returncode)
#         print(e.output)
#         return ""
    
#     #print(len(out))
#     #print(out[:500])
#     return ""


def fetchEntrezAssembliesTableForSpecies(genomeId, kingdomName):
    sleep(requestDelaySeconds)
    
    uri = """https://www.ncbi.nlm.nih.gov/genomes/Genome2BE/genome2srv.cgi?action=GetGenomes4Grid&genome_id=%d&genome_assembly_id=&mode=2&king=%s&flags=1&page=1&pageSize=20""" % (genomeId, kingdomName)
    headers = {'DNT':'1', 'Accept-Encoding':'gzip, deflate, br', 'Accept-Language':'en-US,en;q=0.8', 'User-Agent':'Mozilla/5.0 (X11; Linux x86_64)', 'Accept':'*/*'}
    r = requests.get( uri, headers=headers, stream=True )

    if( r.status_code != 200 ):
        # TODO: Handle other 2xx types?
        print(r.status_code)
        raise Exception("HTTP request failed with code %d" % r.status_code)
    


    return r.content


"""
Fetch entrez genome report HTML for species (identified using genome-id and optional assembly-id)
"""
def fetchEntrezGenomeReportForSpecies(genomeId, assemblyId=None):

    requestUri = None
    if not assemblyId is None:
        requestUri = 'https://www.ncbi.nlm.nih.gov/genome/%d?genome_assembly_id=%d' % (genomeId, assemblyId)
    else:
        requestUri = 'https://www.ncbi.nlm.nih.gov/genome/%d' % genomeId

    sleep(requestDelaySeconds)
    r = requests.get( requestUri )
    if( r.status_code != 200 ):
        raise Exception("Entrez returned HTTP code %d, content %s" % r.status_code, r.text)
    
    return codecs.encode(r.text, 'ascii', 'ignore')

#taxonomicGroupToKingdom = {'Fungi':'Eukaryota', 'Bacteria':'Bacteria', 'Archaea':'Archaea', 'Stramenopiles':'Eukaryota', 'Plants':'Eukaryota', 'Metazoa':'Eukaryota', 'Opisthokonta':'Eukaryota'}
##testTaxIds = (3055, 511145)
#
# """
# Iteration source for (taxId, kingdom) pairs.
# Note: This doesn't return species for which tax-group is not annotated...
# """
# def annotatedKingdomTaxidsSource():
#     allKeys = r.keys("species:taxid:*:tax-group")
#     for key in allKeys:
#         #print(key)
#         taxId = int(key.split(":")[2])
#         taxGroup = getSpeciesTaxonomicGroup(taxId)

#         kingdom = None
#         if taxGroup:
#             kingdom = taxonomicGroupToKingdom[taxGroup]
#             yield (taxId, kingdom)



ncbiTaxa = None

"""
Get kingdom to which a species belongs, in the format recognized by Entrez (see fetchEntrezAssembliesTableForSpecies())
"""
def getKingdomForSpecies(taxId):
    # global ncbiTaxa
    
    # if ncbiTaxa is None: # lazy initialization (not very pythony?)
    #     ncbiTaxa = NCBITaxa()

    # lineage = ncbiTaxa.get_lineage(taxId)
    # assert(lineage[0]==1) # root

    # if lineage[1] == 131567: # cellular organisms
    #     if lineage[2]==2: # Bacteria
    #         return "Bacteria"
    #     elif lineage[2] == 2759: # Eukaryota
    #         return "Eukaryota"
    #     elif lineage[2] == 2157: # Archaea
    #         return "Archaea"

    # raise Exception("Unknown kingdom for lineage %s" % lineage[:3])
    raise Exception("NotImpl")
    

def getTaxonomicGroupForSpecies(taxId):
    global ncbiTaxa

    if ncbiTaxa is None: # lazy initialization (not very pythony?)
        ncbiTaxa = NCBITaxa()

    lineage = ncbiTaxa.get_lineage(taxId)
    assert(lineage[0]==1) # root

    if lineage[1] == 131567: # cellular organisms
        if lineage[2]==2: # Bacteria
            return "Bacteria"
        elif lineage[2] == 2759: # Eukaryota
            return "Eukaryota"
        elif lineage[2] == 2157: # Archaea
            return "Archaea"
        
    raise Exception("Unknown kingdom for lineage %s" % lineage[:3])



def testGettingGenomeAttributes(genomeId, kingdomId):

    # Try getting the assemblies report
    xml = None
    attempt = 1
    while(True):
        try:
            assembliesData = fetchEntrezAssembliesTableForSpecies(genomeId, kingdomId)
            xml = fixMissingImgCloseTags(wrapTableFragmentAsXML(assembliesData))
        except Exception as e:
            sleep(5*requestDelaySeconds)
            attempt += 1
            if attempt > 3:
                raise

        if not xml is None:
            break
    

    # Parse the assemblies report and decide which assembly to use
    (genomeId, assemblyId) = parseNCBIGenomeAssembliesHTML_fetchMainAssembly(xml)
    print((genomeId, assemblyId))

    # Fetch the genome report (for the given genome and assembly)
    report = fetchEntrezGenomeReportForSpecies(genomeId, assemblyId)
    # Return the properties for the genome
    props = parseNCBIGenomeHTML_fetchSummaryReport(report)
    print(props)

    return props
    
def testAll():

    testGettingGenomeAttributes(10796, "Archaea")
    testGettingGenomeAttributes(15,    "Eukaryota")
    testGettingGenomeAttributes(1059,  "Archaea")
    testGettingGenomeAttributes(1030,  "Bacteria")
    testGettingGenomeAttributes(1070,  "Bacteria")
    testGettingGenomeAttributes(1564,  "Archaea")
    testGettingGenomeAttributes(1589,  "Bacteria")
    testGettingGenomeAttributes(1124,  "Bacteria")
    testGettingGenomeAttributes(820,   "Bacteria")
    testGettingGenomeAttributes(1069,  "Bacteria")
    testGettingGenomeAttributes(410,   "Eukaryota")
    testGettingGenomeAttributes(691,   "Bacteria")
    testGettingGenomeAttributes(815,   "Bacteria")
    testGettingGenomeAttributes(416,   "Bacteria")
    testGettingGenomeAttributes(1014,  "Bacteria")

    print("---------------------------------------------")

    totalCount = 0
    envFoundCount = 0
    tempFoundCount = 0
    statsFoundCount = 0

    temps1 = {}
    temps2 = {}
    oxygenReq = {}
    habitat = {}
    salinity = {}
    proteinCount = {}
    gcContent = {}
    genomeSize = {}
    
    for taxId in allSpeciesSource():
        if limitSpecies and taxId not in limitSpecies:
            continue
        
        genomesList = taxIdToGenomeId(taxId)
        if( not genomesList ):
            print("No genome-id found for (taxId=%d), skipping..." % taxId)
            continue

        kingdom = getKingdomForSpecies(taxId)
        genomeId = genomesList[0] # TODO - is this right?
        props = testGettingGenomeAttributes(genomeId, kingdom)

        tempFound = False
        envFound = False
        statsFound = False
        
        if 'Environment:' in props:
            envFound = True
            envprops = props['Environment:']

            if 'TemperatureRange' in envprops:
                tempFound = True
                temps1[taxId] = envprops['TemperatureRange']
                
            if 'OptimumTemperature' in envprops:
                tempFound = True
                temps2[taxId] = envprops['OptimumTemperature']

            if 'OxygenReq' in envprops:
                oxygenReq[taxId] = envprops['OxygenReq']
            
            if 'Salinity' in envprops:
                salinity[taxId] = envprops['Salinity']

            if 'Habitat' in envprops:
                habitat[taxId] = envprops['Habitat']

        else:
            envFound = False

        if 'Statistics:' in props:
            statsFound = True
            stats = props['Statistics:']
            
            if 'protein count' in stats:
                proteinCount[taxId] = stats['protein count']

            if 'GC%' in stats:
                gcContent[taxId] = stats['GC%']

            if 'total length (Mb)' in stats:
                genomeSize[taxId] = stats['total length (Mb)']
        else:
            statsFound = False
                
        totalCount += 1
        if envFound:
            envFoundCount += 1
        if tempFound:
            tempFoundCount += 1
        if statsFound:
            statsFoundCount += 1

    print("TemperatureRange")
    print(temps1)
    print("OptimumTemperature")
    print(temps2)
    print("Salinity")
    print(salinity)
    print("Habitat")
    print(habitat)
    print("OxygenReq")
    print(oxygenReq)
    
    print("ProteinCount")
    print(proteinCount)
    print("GC%")
    print(gcContent)
    print("genomeSize")
    print(genomeSize)
    print("Total: %d\tEnv found: %d\tTemp found: %d\tStats found: %d" % (totalCount, envFoundCount, tempFoundCount, statsFoundCount))

    x = {}
    for k,v in list(temps2.items()):
        if type(v)==type(''):
            if v=='C':
                v = None
            elif v[-1]=='C':
                v = int(v[:-1])
            else:
                v = None
                print("Unknown val %s" % v)
        elif type(v)==type(()):
            if len(v)==2:
                v = (float(v[0])+float(v[1])) / 2
            else:
                v = None
                print("Uknown val %s" % v)
                
        if not v is None:
            x[k] = v
    print(x)

    for taxId, temperature in list(x.items()):
        setSpeciesProperty( taxId, 'optimum-temperature', '%g'%temperature, "entrez", overwrite=False )
                
    for taxId, tempRange in list(temps1.items()):
        setSpeciesProperty( taxId, 'temperature-range',    tempRange,       "entrez", overwrite=False )

    for taxId, val in list(salinity.items()):
        if val=='Unknown':
            continue
        setSpeciesProperty( taxId, 'salinity',    val,       "entrez", overwrite=False )

    for taxId, val in list(habitat.items()):
        if val=='Unknown':
            continue
        setSpeciesProperty( taxId, 'habitat',    val,       "entrez", overwrite=False )

    for taxId, val in list(oxygenReq.items()):
        if val=='Unknown':
            continue
        setSpeciesProperty( taxId, 'oxygen-req',    val,       "entrez", overwrite=False )
        
    for taxId, val in list(proteinCount.items()):
        setSpeciesProperty( taxId, 'protein-count',    val,       "entrez", overwrite=False )

    for taxId, val in list(gcContent.items()):
        if( val > 90 or val < 10 ):
            continue
        
        if( setSpeciesProperty( taxId, 'gc-content',    "%g"%val,       "entrez", overwrite=False ) ):
            print("[gc-content (taxid=%d) -> %g]" % (taxId, val))

    for taxId, val in list(genomeSize.items()):
        setSpeciesProperty( taxId, 'genome-size-mb',    "%g"%val,       "entrez", overwrite=False )
        
    return 0



    genomeIdentifiers = taxIdToGenomeId(3055) # Obtain genome-ids for this tax-id
    for genomeId in genomeIdentifiers:
        
        report = fetchEntrezGenomeReportForSpecies(genomeId)
        props = parseNCBIGenomeHTML_fetchSummaryReport(report)
        print(props)
    #return 0
    #------------
    #for fn in ("NCBI_genome_1030.html", "NCBI_genome_1070.html", "NCBI_genome_1347.html"):
    #    with open(fn, "r") as f:
    #        print("Testing %s..." % fn)
    #        props = parseNCBIGenomeHTML_fetchSummaryReport(f.read())
    #        print(props)

    for fn in ("NCBI_genomes_report_15_table.txt",):
        with open(fn, "r") as f:
            print("Testing %s..." % fn)
            (genomeId, assemblyId) = parseNCBIGenomeAssembliesHTML_fetchMainAssembly(
                fixMissingImgCloseTags(wrapTableFragmentAsXML(f.read()))
            )
            print((genomeId, assemblyId))
    
    return 0

if __name__=="__main__":
    import sys
    sys.exit(testAll())
