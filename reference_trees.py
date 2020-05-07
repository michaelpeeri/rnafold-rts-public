from builtins import zip
from builtins import str
from builtins import range
from past.builtins import basestring
from builtins import object
import re
from collections import Counter
import xml.etree.ElementTree as ET
from ete3 import NCBITaxa, Tree, TreeStyle, NodeStyle, TextFace, faces, AttrFace, PhyloTree
from Bio import Entrez, AlignIO, Phylo
from rate_limit import RateLimit
from ncbi_taxa import ncbiTaxa
#from fuzzy_text_matching import sw
import pandas as pd



# configuration
# nmicrobiol201648_s4             = "./data/nmicrobiol201648-s4.txt.fixed"
# nmicrobiol201648_s5             = "./data/nmicrobiol201648-s5.txt"    # =Alignment
# nmicrobiol201648_s6             = "./data/nmicrobiol201648-s6.txt"    # =Tree
# nmicrobiol201648_s6_PATHd8      = "./data/nmicrobiol201648-s6.txt.nw.PATHd8.out.d8.nw"    # =Tree
# nmicrobiol201648_s8             = "./data/nmicrobiol201648-s8.txt"
# itol_newick                     = "./data/itol_newick.txt"
# Entrez.email                    = "mich1@post.tau.ac.il"
# unhandledXMLsFile               = "reference_tree.unhandled_xml.txt"
# nodeIdentifiersMappingTable_csv = "reference_trees.hug.identifiers.csv"
# nodeIdentifiersMappingTable_xls = "reference_trees.hug.identifiers.xlsx"
# nodeIdentifiersMappingTable_with_testing_csv = "reference_trees.hug.identifiers.with_testing.csv"
# nodeIdentifiersMappingTable_with_inclusion_csv = "reference_trees.hug.included_for_itol.csv"
# alignmentIdentifiersToTreeIdentifiersMappingTable_csv = "reference_trees.hug.alignment_ids.csv"


##taxonItemsToIgnore = frozenset(("Unclassified", "Incertae", "Candidate", "candidate", "Candidatus", "incertae", "sedis"))
#taxonItemsToIgnore  = frozenset(("Unclassified", "Incertae",                                         "incertae", "sedis"))

reNodeSupport = re.compile('[[][^]]*[]]')
def stripTreeInternalSupport(tree):
    return re.sub(reNodeSupport, '', tree)

rl = RateLimit(15)


def getContainingSpeciesLevelTaxon(taxId):
    lineage = ncbiTaxa.get_lineage(taxId)
    ranks = ncbiTaxa.get_rank(lineage)

    for taxon in reversed(lineage):
        rank = ranks[taxon]
        if rank == "no rank":
            continue
        elif rank == "subspecies":
            continue
        elif rank == "forma":
            continue
        elif rank == "varietas":
            continue
        elif rank == "species":
            return taxon
        else:
            #print("--"*50)
            #print(taxId)
            #print("Lineage: %s" % lineage)
            #print("Ranks: %s" % ranks)
            #print(ncbiTaxa.get_taxid_translator(lineage))
            return None


# class FuzzyTaxonMatcher(object):
#     def __init__(self, requiredQueryIdentity=0.6):
#         self._requiredQueryIdentity = requiredQueryIdentity
#         self._db = ncbiTaxa.db # use NCBITaxa's sqlite connection

#     def match(self, query, scope, verbose=False):
#         out = []
#         for taxId, ncbiMatch in self._db.execute('select taxid, spname from species where spname like "%s%%";' % scope).fetchall():  # TODO where rank in (111,222,333)
#             match = sw.align( ncbiMatch, query)
#             out.append( (taxId, ncbiMatch, match, match.score) )

#         out.sort(key = lambda x:-(x[3]) )
#         if verbose:
#             for x in out[:3]:
#                 print("--"*20)
#                 print("%d %s" % (x[0], x[1]))
#                 print( "score: %d matches: %d" % (x[2].score, x[2].matches))
#                 x[2].dump()
#         else:
#             #out[0][2].dump()
#             pass


#         if not out:
#             return None

#         if out[0][2].matches < len(query)*self._requiredQueryIdentity:
#             return None

#         if len(out)>1 and not ( ( out[0][3] > out[1][3]) and (out[0][2].matches > out[1][2].matches ) ):
#             return None

#         return (out[0][0], out[0][1], float(out[0][2].matches)/len(query))


        
        
        
def pruneTree(tree, keepNodes=None, keepTaxIds=None, saveTreeAs=None):

    # Collect all taxids to keep (based on keepNodes and keepTaxIds):
    _allTaxIdsToKeep = set()
    # first, if we got nodes, extract the taxid from each one
    if( not keepNodes is None ):
        for x in keepNodes:
            if not x.taxId is None:
                _allTaxIdsToKeep.add(x.taxId)

    # second, if we got taxids, also add them to the list of kept taxids
    if( not keepTaxIds is None ):
        _allTaxIdsToKeep = _allTaxIdsToKeep.union( frozenset(keepTaxIds) )

    print("Pruning, will keep %d nodes" % len(_allTaxIdsToKeep))

    for node in tree.traverse(strategy='postorder'):
        childTaxIds = set()
        
        for x in node.get_leaves():
            try:
                if not x.taxId is None:
                    childTaxIds.add(x.taxId)
            except AttributeError as e:
                pass
        #print(len(childTaxIds))

        if not _allTaxIdsToKeep.intersection(childTaxIds):
            node.detach()
            #node.delete()
            #print(".")

    nn = []
    for node in tree.traverse():
        if node.is_leaf():
            nn.append(node)

    print("nn = %d" % len(nn))
    print(len(tree))
    tree.prune(nn, preserve_branch_length=True)
    print(len(tree))

    print("Expected: %d  Actual: %d" % (len(_allTaxIdsToKeep), len(tree)))
    if not saveTreeAs is None:
        tree.write(format=1, outfile=saveTreeAs)

    #assert(len(tree) <= len(taxIds))
    return tree
          

def pruneTree2(tree, keepNodes):
    tree.prune(keepNodes)
            

def pruneTreeByTaxonomy(tree, parentTaxonIdToKeep):
    allKeepTaxIds = set()

    # Collect all child taxons whose lineage includes the specified parent
    for node in tree.traverse():
        if node.is_leaf():
            leafTaxId = getattr(node, "taxId", None)
            if (not leafTaxId is None):
                lineage = frozenset(ncbiTaxa.get_lineage(node.taxId))

                if( parentTaxonIdToKeep in lineage ):
                    allKeepTaxIds.add( node.taxId )

    return pruneTree(tree, keepTaxIds=allKeepTaxIds )

def getTaxidsFromTree(tree):
    out = []
    for node in tree.traverse():
        if node.is_leaf():
            if (not node.taxId is None):
                out.append(node.taxId)
    return out


def extendTreeWithSpecies( tree, additionalSpecies, limitTaxonomy=None ):
    
    # Collect the tax-ids of all species included in the tree
    realTreeSpecies = frozenset(getTaxidsFromTree(tree))

    for taxId in additionalSpecies:
        if taxId in realTreeSpecies: continue # this species is included in the tree, no need to append it

        if not limitTaxonomy is None:
            lineage = frozenset(ncbiTaxa.get_lineage(taxId))
            if( not limitTaxonomy in lineage ):   # this species does not belong to the specified taxon
                continue

        # Create a compatible node for this species, and add it to the tree as an outgroup
        newNode = tree.add_child( name=str(taxId), dist=0.1 )
        newNode.add_features( taxId = taxId, label = taxId, dummyTopology = True )
        
    return tree
    

def readTranslationMap():
    treeNodeIdentifiersDf = pd.read_csv(nodeIdentifiersMappingTable_with_testing_csv, dtype={'NodeLabel': 'string', 'DBIdentifier': 'string', 'DBIdentifierType': 'category', 'TaxId': 'int32'} )

    out = {}
    for row in treeNodeIdentifiersDf.itertuples():
        label = row.NodeLabel
        taxId = row.TaxId
        if not taxId is None and taxId != 0:
            out[label] = taxId
            
    return out

            

def inferTaxIdForLabel(label):
    # Try to match the species name with the NCBI taxonomy
    # The node names contain concatenated taxon names and we don't know where the species name starts.
    # We will collect candidate strings (based on the criteria below) and choose the candidate that has a match in NCBI.
    #
    # See also: FuzzyMatch in R 'evobiR' package (https://cran.r-project.org/web/packages/evobiR/index.html)
    #
    items = label.split("_") # "items" (words) are separated by '_'

    candidates = []
    for i in range(len(items)-1, 0, -1): # process suffixes (strings containing last N items), from shortest to longest
        candidate = " ".join(items[i:])

        # Filter candidates that are too short or too long to be a binomial species name (with optional strain description)
        if len(candidate) < 8:
            continue # skip this suffix

        if len(candidate) > 100:
            break # skip this and longer suffixes

        if items[i] in taxonItemsToIgnore:
            continue

        if candidate.find("ncertae sedis") > -1:
            continue

        candidates.append( (i,candidate,-1) )

        # For each suffix, also consider its prefixes as candidates. This is bacause the species names sometimes include additional strain identifiers that are not included in NCBI
        if i<=len(items)-2 and items[i] and items[i+1] and items[i][0].isupper() and items[i+1][0].islower(): # require that the first item start with a capital, and the second item start with a lower-case letter (e.g., "Eschericia coli")
            for j in range(i+1,len(items)-1):
                candidate = " ".join(items[i:j+1])

                candidates.append( (i,candidate,j+1) )


    if " ".join(items).find("Curtiss") > -1:
        print("="*50)
        print(items)
        print(candidates)
        print("="*50)

    # Sort candidates by priority - from longest to shortest
    candidates = sorted(candidates, key=lambda x:-len(x[1]))

    # Finished collecting candidates; match them in NCBI
    # Note: matching all names in a single call is much faster
    ncbiMatches = ncbiTaxa.get_name_translator([x[1] for x in candidates])

    speciesStartItem = None
    speciesLevelTaxon = None
    matchingName = None
    matchingTaxId = None

    # Find the first (i.e., longest) match
    for candidateStartItem, candidateName, _ in candidates:
        if candidateName in ncbiMatches:
            candidateTaxId = ncbiMatches[candidateName][0] 

            speciesLevelTaxon = getContainingSpeciesLevelTaxon( candidateTaxId )
            if speciesLevelTaxon is None: # Make sure this taxon is not above species rank
                continue

            # select this candidate
            matchingName = candidateName
            matchingTaxId = candidateTaxId
            speciesStartItem = candidateStartItem # remember at what item the selected species starts
            break
    return (matchingTaxId, matchingName)
            
        
def pruneReferenceTree_Nmicrobiol201648(taxa):
    treefmt = None
    
    with open(nmicrobiol201648_s6_PATHd8, "r") as f:
    #with open(itol_newick, "r") as f:
        treefmt = f.read()

    # Strip internal support, and parse the resulting string
    tree = PhyloTree(stripTreeInternalSupport(treefmt))

    #ts.show_leaf_name = True
    #ts.layout_fn = nodeLayoutWithTaxonomicNames

    unmatched = []
    matched = []

    #xxx = set([1287680, 115713, 203267, 188937, 4781, 187420, 243230, 130081, 227882, 228908, 227377, 224308, 5693, 345663, 208964, 224325, 1116230, 243273, 213585, 64091, 45670, 1069680, 1397361, 280699, 1047168, 284811, 284812, 46234, 418459, 214684, 262768, 243365, 273063, 511145, 176299, 272557, 272558, 402612, 283166, 223926, 163003, 559292, 1041607, 1183438, 2769, 122586, 273116, 593117, 192222, 1574623, 243159, 160490, 212717, 272623, 272631, 272632, 63737, 272634, 1341181, 1125630, 99287, 27923, 400667, 269084, 257314, 96563, 300852, 4927, 381764, 242507, 65357, 104782, 336722, 190304, 882, 347515, 353152, 83332, 93061, 194439, 1223560, 267671, 196164, 1245935, 449447, 420778, 195522, 556484, 5061, 391623, 70601, 85962, 272844, 259536, 272633, 220668, 169963, 295405, 237561, 407035, 997884, 1432061, 1010810, 562, 1010800])

    labelToTaxId = readTranslationMap()
        
    numProcessed = 0
    numMatched = 0
    # Annotate tree leaves
    for node in tree.traverse():
        if node.is_leaf():
            #items = node.name.split("_") # "items" (words) are separated by '_'

            #for i,x in enumerate(items):
            #    if x.startswith("Submit") or x.startswith("submit"):
            #        print("Removing submission note on node: %s" % items)
            #        items = items[:i]
            #        break
            
            matchingName = None
            matchingTaxId = None
            #print("---------------------------")
            #print(n)


            # Check if the label has a mapping in the id-conversion table
            matchingTaxId = labelToTaxId.get(node.name)

            #if not matchingTaxId is None:
            #    matchingName = ncbiTaxa.get_taxid_translator((matchingTaxId,))[matchingTaxId]
                    
            # Did we find a match for this leaf?
            #if not matchingName is None: # match found
            if not matchingTaxId is None:
                node.label = node.name
                node.name = str(matchingTaxId)
                #node.matchingName
                #print("<%s>" % node.name)

                #lineageItems = [x for x in items[:speciesStartItem] if not x in taxonItemsToIgnore]
                lineageItems = ncbiTaxa.get_lineage(matchingTaxId)
                
                # TODO - Fix lineageItems ?
                node.add_features(taxId = matchingTaxId, lineageItems = lineageItems)
                #node.add_features(taxId = matchingTaxId)
                #if matchingTaxId != speciesLevelTaxon:
                #    node.add_features(speciesLevelTaxon=speciesLevelTaxon)

                matched.append("%s [%d]" % (node.label, matchingTaxId))


                #print("--"*20)
                #print(matchingName)
                #print(items[:speciesStartItem])
                numMatched += 1
            else: # no match found
                unmatched.append(node.name)
                node.name = "n/a"

            numProcessed += 1
            if (rl()):
                print("(processed %d matched %d)" % (numProcessed, numMatched))

            #if(numProcessed>1000):
            #    break


    # Save unmatched names to file (for examination)
    with open("unmatched_names.txt", "w") as f:
        f.writelines(["%s\n" % x for x in sorted(unmatched)])

    # Save unmatched names to file (for examination)
    with open("matched_names.txt", "w") as f:
        f.writelines(["%s\n" % x for x in sorted(matched)])

    print("//"*30)
    print("//"*30)
    print("//"*30)
    print("//"*30)

    outer = {}

    # Try to annotate non-leaf nodes with common taxonomic group
    for node in tree.traverse(strategy='postorder'): # children first
        if node.is_leaf():
            continue

        a = []
        for c in node.children:
            try:
                a.append( c.lineageItems )
            except AttributeError:
                pass


        print(">>" * 20)
        print(len(a))
        print(a)
        
        if a:
            out = None
            
            if len(a)==2:
                out = []
                #if a[0][-1]=="Anabaena" or a[1][-1]=="Anabaena":
                #    print(">"*50)
                #    print(a)
                #    print(">"*50)
                    
                for u,v in zip(a[0], a[1]):
                    if u==v:
                        out.append(u)
                    else:
                        break
            elif len(a)==1:
                #if a[0][-1]=="Anabaena":
                #    print(">"*50)
                #    print(a)
                #    print(">"*50)
                    
                out = a[0]
            else:
                assert(False)

            if out:
                #print("out = %s" % out)
                print("//"*20)
                print(out[-1])
                node.add_features(lineageItems = out, testK = out[-1])

                outer[out[-1]] = id(node)

                print(">>> %s" % out[-1])

                if out[-1]==2:
                    print("- - "*5)
                    print(a)
                    print("- - "*5)
                
            #    #if out[-1] == "Nostocaceae":
            #        #print(node.name)
            #        #print(a)
            #        #pass
                    
            #else:
            #    print("*-"*20)
            #    print(node.name)
            #    print(a)
                
    for node in tree.traverse(strategy='postorder'): # children first
        if node.is_leaf():
            continue

        try:
            l = node.testK
            if not l is None:
                if outer[l]==id(node):
                    node.add_features(testL = l)

        except AttributeError as e:
            pass
        


    print("//"*30)
    print("//"*30)
    print("//"*30)
    print("//"*30)
        
    # Now we have our annotated reference phylogenetic tree

    # Get our target list of species to appear on the final tree
    #taxa = getSpeciesToInclude()
    allNames = ncbiTaxa.get_taxid_translator(taxa)

    
    #taxa.append(1906157)
    #taxa.append(251221)

    #print(ncbiTaxa.get_rank(taxa))

    for x in (45157,4896,44056):
        print(x)
        print(tree.search_nodes(taxId=x))
        
    
    f = set()
    fnodes = []
    notf = set()
    for x in taxa:
        print("=="*20)
        print("Searching for %d" % x)
        found = tree.search_nodes(taxId=x)
        
        if found:
            f.add(x)
            fnodes.append(found[0])
            print("Exact match found")
        else:
            containingSpeciesLevelTaxon = getContainingSpeciesLevelTaxon(x)
            if x != containingSpeciesLevelTaxon:
                found = tree.search_nodes(taxId=containingSpeciesLevelTaxon)

                if not found:
                    found = tree.search_nodes(speciesLevelTaxon=containingSpeciesLevelTaxon)

                # TODO - CONTINUE HERE
                #if not found:
                #    found = tree.search_nodes(
                    
                if found:
                    f.add(x)
                    fnodes.append(found[0])
                    print("Found")
                else:
                    print("Not found at all...")
                        
        #elif ncbiTaxa.get_rank([x])[x] == 'no rank':
        #    parent = ncbiTaxa.get_lineage(x)[-2]
        #    found = tree.search_nodes(taxId=parent)
        #    if found:
        #        f.add(x)
        #        fnodes.append(found[0])
                

        if x not in f:
            #print("--"*50)
            #print("TaxId not found: %d" % x)
            #print("Name: %s" % allNames[x])
            containingSpeciesLevelTaxon = getContainingSpeciesLevelTaxon(x)
            #print("Species TaxId: %d %s" % (containingSpeciesLevelTaxon, "" if x==containingSpeciesLevelTaxon else "***"))
            
            lineage = ncbiTaxa.get_lineage(x)
            #print("Lineage: %s" % lineage)
            names = ncbiTaxa.get_taxid_translator(lineage)
            #print(names)

            for y in reversed(lineage):
                name = names[y]
                res = bool(tree.search_nodes(testK = name))
                #print("%s: %s" % (name, res))
            notf.add(x)
            
    print("Found (%d): %s" % (len(f), f))
    print(ncbiTaxa.get_rank(list(f)))

    print("Couldn't find (%d): %s" % (len(notf), notf))
    print(list(ncbiTaxa.get_taxid_translator(list(notf)).values()))
    print(len(fnodes))

    tree2 = tree.copy()

    print("Before pruning: %d" % len(tree2))
    if fnodes:
        #tree2.prune(fnodes, preserve_branch_length=True)
        pruneTree(tree2, fnodes)
    print("After pruning: %d" % len(tree2))

    return (tree, tree2)

"""
Parse Supplamentary Table 1 from Hug et. al., containing external DB (NCBI or JGI) identifiers for samples used in tree nodes.
"""
def readIdentifiersTable():
    from csv import reader
    count = 0

    identifiers = {}
    
    with open( nmicrobiol201648_s4, 'rU') as csvfile:  # This is a mac file with '\r' line endings
        
        for row in reader( csvfile, delimiter='\t' ):
            if len(row) < 5:
                continue
            
            (stringIdentifier, _, ncbiIdentifier, JGIIdentifier, _) = row

            if( not stringIdentifier ):
                continue

            if not (ncbiIdentifier or JGIIdentifier):
                print("Warning: skipping sample missing any identifiers: %s" % stringIdentifier)
                continue

            count += 1

            if (ncbiIdentifier and JGIIdentifier):
                print("Warning: sample has multiple identifiers: %s" % stringIdentifier)

            if ncbiIdentifier:
                identifiers[stringIdentifier] = (ncbiIdentifier, 'N')
            elif JGIIdentifier:
                identifiers[stringIdentifier] = (JGIIdentifier, 'J')
            else:
                assert(False)
                
    print("Read %d rows" % count)
    return identifiers






treeNodeIdentifiersDf = pd.DataFrame({'NodeLabel': pd.Series([], dtype='str'),
                                      'DBIdentifier': pd.Series([], dtype='str'),
                                      'DBIdentifierType': pd.Categorical([]),
                                      'TaxId': pd.Series([], dtype='int')})


badTaxIdsForBiosamples = frozenset((646099, 9606))  # Ignore biosamples coming from human or human metagenomes (since we are looking for individual species to assign to tree nodes). Why are these even included?
#
# TODO - add filter to ignore samples assigned to other metagenomes (under "unclassified [12908] -> metagenome [408169]" in NCBITaxon hierarchy)
#
alreadyFound = set()
failedBiosampleIdentifiers = []
"""
Return taxid for species-specific NCBI Biosample IDs; Try to reject multiple-species or metagenomic samples.
'sampleIdentifier' may be a string or sequence of strings
"""
def translateNCBIBioSampleIdentifier(sampleIdentifier, identifierClass):

    # DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #
    #if identifierClass=="SAMN":
    #    return None
    # DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #### DEBUG ONLY #

    #print(sampleIdentifier)
    handle = Entrez.efetch(db="biosample", id=sampleIdentifier, retmode="xml")  # id can be a string or sequence of strings
    tree = ET.parse(handle)
    handle.close()

    out = []

    for sample in tree.findall("./BioSample"):
        accession = sample.get("accession")

        taxId = None
        
        organism = sample.findall("./Description/Organism")
        if len(organism) == 1:  # reject taxid if there are multiple annotations
            taxId = int(organism[0].get("taxonomy_id"))

        if taxId in badTaxIdsForBiosamples:
            taxId = None

        # Detect multiply-occurring taxId and issue warning
        if not taxId is None:
            if taxId in alreadyFound:
                print("Warning: multiple occurence of taxId %d detected in record %s" % (taxId, sampleIdentifier))
            else:
                alreadyFound.add(taxId)
 
        if taxId is None:
            failedBiosampleIdentifiers.append(sampleIdentifier)
        out.append(taxId)

    # Check the returned value, according to the number of items requested
    if isinstance(sampleIdentifier, basestring):
        assert(len(out)==1)
    elif isinstance(sampleIdentifier, Iterable):
        assert(len(out)==len(sampleIdentifier))
    else:
        assert(False)
    
    return out




failedSequenceIdentifiers = []
# <eLinkResult>
#     <LinkSet>
#         <DbFrom>nuccore</DbFrom>
#         <IdList>
#             <Id>851302729</Id>
#         </IdList>
#         <LinkSetDb>
#             <DbTo>taxonomy</DbTo>
#             <LinkName>nuccore_taxonomy</LinkName>
#             <Link>
#                 <Id>1343739</Id>
#             </Link>
#         </LinkSetDb>
#     </LinkSet>
# </eLinkResult>
def translateNCBISequenceIdentifier(seqIdentifier, identifierClass):
    #record = Entrez.read(Entrez.elink(dbfrom="pubmed", id=pmid))
    handle = Entrez.elink(db="taxonomy", dbfrom="nucleotide", id=seqIdentifier, retmode="xml")
    tree = ET.parse(handle)
    handle.close()

    taxId = None
    
    taxIdNode = tree.findall("./LinkSet/LinkSetDb/Link/Id")
    if not taxIdNode is None and len(taxIdNode)==1:
        taxId = int(taxIdNode[0].text)
    else:
        print("XX[seq] %s" % seqIdentifier)
        
        with open(unhandledXMLsFile, "w+") as f:
            f.write("\n%s\n" % "--"*20)
            f.write("ID[seq]: %s\n" % seqIdentifier)
            f.write(ET.tostring(tree.getroot()))
            f.write("\n")
            f.flush()
            
        failedSequenceIdentifiers.append(seqIdentifier)

    return [taxId]




# <RecordSet><DocumentSummary uid="47111">
#     <Project>
#         <ProjectID>
#             <ArchiveID accession="PRJNA47111" archive="NCBI" id="47111" />
#         </ProjectID>
#         <ProjectDescr>
#             <Name>Bigelowiella natans CCMP2755</Name>
#             ...
#         </ProjectDescr>
#         <ProjectType>
#             <ProjectTypeSubmission>
#                 <Target capture="eWhole" material="eGenome" sample_scope="eMonoisolate">
#                     <Organism species="227086" taxID="753081">
#                         <OrganismName>Bigelowiella natans CCMP2755</OrganismName>
#                         <Strain>CCMP2755</Strain>
#                         <Supergroup>eEukaryotes</Supergroup>
#                         <GenomeSize units="Kb">94700.000000</GenomeSize>
#                     </Organism>
failedProjectIdentifiers = []
def translateNCBIBioProjectIdentifier(prjIdentifier, identifierClass):
    handle = Entrez.efetch(db="bioproject", id=prjIdentifier, retmode="xml")
    tree = ET.parse(handle)
    handle.close()

    
    taxId = None
    
    organismNode = tree.findall("./DocumentSummary/Project/ProjectType/ProjectTypeSubmission/Target/Organism")
    if not organismNode is None and len(organismNode)==1:
        taxId = int(organismNode[0].get("taxID"))
        speciesTaxId = int(organismNode[0].get("species"))  # ignored...
    else:
        print("XX[prj] %s" % prjIdentifier)

        with open(unhandledXMLsFile, "w+") as f:
            f.write("\n%s\n" % "--"*20)
            f.write("ID[prj]: %s\n" % prjIdentifier)
            f.write(ET.tostring(tree.getroot()))
            f.write("\n")
            f.flush()
        
        failedProjectIdentifiers.append(prjIdentifier)


    return [taxId]
    

reNCBIIdentifier = re.compile("([A-Z]+)_?\w+")

x = Counter()

def translateNCBIIdentifier(dbIdentifier):
    #from random import randint   # DEBUG ONLY
    # Determine the accession class
    # https://www.ncbi.nlm.nih.gov/books/NBK21091/table/ch18.T.refseq_accession_numbers_and_mole/?report=objectonly
    #
    # https://www.ncbi.nlm.nih.gov/Sequin/acc.html
    #
    # Accession classes in Hug et. al. supp. table 1:
    # Counter({'SAMN': 1475, 'NZ': 634, 'NC': 596, 'PRJNA': 171, 'CP': 21, 'SAMEA': 6, 'FP': 5, 'PRJEA': 5, 'SAMD': 5, 'PRJDA': 3, 'EKD': 3, 'FN': 2, 'AE': 2, 'AP': 2, 'PRJEB': 2, 'AUYT': 1, 'BA': 1, 'AWSN': 1, 'AYLK': 1, 'EKE': 1, 'AYLL': 1, 'APKF': 1, 'CAFG': 1, 'JYIM': 1, 'ASAI': 1, 'ATBP': 1, 'AKKQ': 1, 'ASPK': 1, 'AGCY': 1, 'APGO': 1, 'ASZN': 1, 'AQTX': 1, 'JWKR': 1, 'AYLI': 1, 'ADVD': 1, 'CA': 1, 'L': 1, 'JRFF': 1, 'AGNT': 1, 'JWKP': 1, 'AXWL': 1, 'JWKN': 1, 'JWKQ': 1, 'JWKT': 1, 'JWKU': 1, 'JWKV': 1, 'JWKW': 1, 'LSSB': 1, 'ARQD': 1, 'JWKS': 1, 'JWKX': 1, 'JWKY': 1, 'JWKZ': 1, 'JQJA': 1, 'DP': 1, 'KT': 1, 'AXCJ': 1, 'ARWQ': 1, 'JWKO': 1})
    #
    match = reNCBIIdentifier.match(dbIdentifier)
    ncbiIdentifierClass = None
    if match is None:
        x.update(("-",))
        return None
    else:
        ncbiIdentifierClass = match.group(1)
        x.update((ncbiIdentifierClass,))

    if ncbiIdentifierClass.startswith("SAM"):
        #if randint(0, 9)>0:
        #    return None
        #else:
        return translateNCBIBioSampleIdentifier( dbIdentifier, ncbiIdentifierClass )[0]
        
    elif ncbiIdentifierClass.startswith("PRJ"):
        return translateNCBIBioProjectIdentifier( dbIdentifier, ncbiIdentifierClass )[0]
    
    else:
        #if randint(0, 9)>0:
        #    return None
        #else:
        return translateNCBISequenceIdentifier( dbIdentifier, ncbiIdentifierClass )[0]
        


allJGIIds = []
def translateJGIIdentifier(dbIdentifier):
    # TODO - IMPL THIS (but only 56/~3000 species, and almost all of these are not at the species level)
    allJGIIds.append(dbIdentifier)
    return None


def storeCorrelatedIdentifier( treeNodeIdentifier, dbIdentifierValue, dbIdentifierSource, taxId):
    #fout.write("%s\t%s\t%s\t%s\n" % (treeNodeIdentifier, dbIdentifierValue, dbIdentifierSource, "" if taxId is None else str(taxId)))
    global treeNodeIdentifiersDf
    
    treeNodeIdentifiersDf = treeNodeIdentifiersDf.append(
        pd.DataFrame({'NodeLabel': pd.Series([treeNodeIdentifier], dtype='str'),
                      'DBIdentifier': pd.Series([dbIdentifierValue], dtype='str'),
                      'DBIdentifierType': pd.Categorical([dbIdentifierSource]),
                      'TaxId': pd.Series([0 if taxId is None else taxId], dtype='int')}) )

topLevelTaxons = Counter()
def isTaxonParticularCellularSpecies(taxId):
    assert(not taxId is None)

    lineage = ncbiTaxa.get_lineage(taxId)
    ranks = ncbiTaxa.get_rank(lineage)

    assert(lineage[0]==1)     # 'Root' node of taxonomy

    # Reject samples that belong to viruses, or to the "Unspecified" subtree of the taxonomy (e.g., metagenomic samples)
    if lineage[1] != 131567:  # Cellular organism
        return False

    topLevelTaxons.update([lineage[1]])  # Keep track of encountered top-level taxons

    # Reject samples that are not specified at the species level
    if "species" in list(ranks.values()):
        return True
    else:
        return False


def prepareTranslationMap():
    identifiers = readIdentifiersTable()
    
    numProcessed = 0
    numFound = 0

    for treeNodeIdentifier, dbIdentifier in list(identifiers.items()):
        (dbIdentifierValue, dbIdentifierSource) = dbIdentifier

        taxId = None

        if dbIdentifierSource=='N':    # NCBI
            taxId = translateNCBIIdentifier( dbIdentifierValue )
            #if not taxId is None:
            #    print(taxId)

        elif dbIdentifierSource=='J':   # JGI
            taxId = translateJGIIdentifier( dbIdentifierValue )

        else:
            assert(False)

        if (not taxId is None) and (not isTaxonParticularCellularSpecies(taxId)):
            taxId = None

        numProcessed += 1
        if not taxId is None:
            numFound += 1

        # TODO: store taxId
        storeCorrelatedIdentifier( treeNodeIdentifier, dbIdentifierValue, dbIdentifierSource, taxId )

        if( rl() ):
            print("%d processed, %d found" % (numProcessed, numFound))


    # Done
    # Save results to file
    treeNodeIdentifiersDf.to_csv(  nodeIdentifiersMappingTable_csv )
    treeNodeIdentifiersDf.to_excel(nodeIdentifiersMappingTable_xls, sheet_name='NodeTaxonMapping')

    print("Identifier classes counts:")
    print(x) 
    print("Failed Sequence identifiers:")
    print(failedSequenceIdentifiers)
    print("Failed Biosample identifiers:")
    print(failedBiosampleIdentifiers)
    print("Failed Project identifiers:")
    print(failedProjectIdentifiers)
    print("JGI Ids:")
    print(allJGIIds)

    print("Top-level taxons:")
    print(topLevelTaxons)


def taxonomicTreeDistance(taxId1, taxId2):
    if taxId1==taxId2:
        return 0
    assert(taxId1>0)
    assert(taxId2>0)
    
        
    

    lineage1 = ncbiTaxa.get_lineage(taxId1)
    lineage2 = ncbiTaxa.get_lineage(taxId2)

    lastMatch = 0
    for u,v in zip(lineage1, lineage2):
        if u != v:
            break
        else:
            lastMatch += 1
            
    return max(len(lineage1),len(lineage2)) - lastMatch

def testTranslationMap():
    treeNodeIdentifiersDf = pd.read_csv(nodeIdentifiersMappingTable_csv, dtype= {'NodeLabel': 'string', 'DBIdentifier': 'string', 'DBIdentifierType': 'category', 'TaxId': 'int32'} )

    #treeNodeIdentifiersDf['InferredTaxId'] = 0

    inferredTaxIds = pd.Series([], dtype='int32')
    inferredTaxIdDists = pd.Series([], dtype='int32')

    i = 0
    for row in treeNodeIdentifiersDf.itertuples():
        inferredTaxId =  inferTaxIdForLabel(row.NodeLabel)[0]
        print("%s %s" % (row.TaxId, inferredTaxId))
        inferredTaxIds[i] = 0 if inferredTaxId is None else inferredTaxId

        dist = None
        if row.TaxId==0 or inferredTaxId is None:
            dist = 99
        else:
            dist = taxonomicTreeDistance(row.TaxId, inferredTaxId)
        inferredTaxIdDists[i] = dist
        i += 1

    treeNodeIdentifiersDf['InferredTaxId']     = inferredTaxIds
    treeNodeIdentifiersDf['InferredTaxIdDist'] = inferredTaxIdDists
    
    #treeNodeIdentifiersDf.assign( Match = (TaxId==InferredTaxId) )

    treeNodeIdentifiersDf.to_csv( nodeIdentifiersMappingTable_with_testing_csv )
    
    return 0


"""
Write table showing all tree nodes, with value indicating for each node whether it has been imported into our DB.
This report is can be used as an "external dataset" in iTOL viewer, *if the appropriate header is prepended*.
"""
def outputNodeExistenceInRnafoldDB():
    from data_helpers import getSpeciesName
    treeNodeIdentifiersDf = pd.read_csv(nodeIdentifiersMappingTable_csv, dtype= {'NodeLabel': 'string', 'DBIdentifier': 'string', 'DBIdentifierType': 'category', 'TaxId': 'int32'} )

    existenceStatuses = pd.Series([], dtype='int32')

    values = Counter()
    
    i=0
    for row in treeNodeIdentifiersDf.itertuples():
        isIncluded = not (getSpeciesName(row.TaxId) is None)
        existenceStatuses[i] = 1 if isIncluded else 0

        values.update((isIncluded,))
        
        i += 1

    treeNodeIdentifiersDf['Included']     = existenceStatuses

    del treeNodeIdentifiersDf['DBIdentifier']
    del treeNodeIdentifiersDf['DBIdentifierType']
    del treeNodeIdentifiersDf['TaxId']
    del treeNodeIdentifiersDf['Unnamed: 0']
    
    treeNodeIdentifiersDf.to_csv( nodeIdentifiersMappingTable_with_inclusion_csv, index=False )

    print("Output values summary: %s" % values)
    
    return 0

def removeUninformativeTerms(name):
    #n2 = name[:]
    name = name.replace("_CPR_", "_")
    name = name.replace("_CP_", "_")
    name = name.replace("_Unclassified_", "_")
    name = name.replace("_unclassified_", "_")
    name = name.replace("_Unclassfied_", "_")
    name = name.replace("_Doykabacteria", "_Dojkabacteria")
    #if n2 != name:
    #    print("%s -> %s" % (n2, name))
    return name

# def findBestMatches(n1, n2):
#     bestMatches = {}
#     for a in n1:

#         best = None
#         best_b = None
        
#         for b in n2:
#             match = sw.align( removeUninformativeTerms(a), removeUninformativeTerms(b))
#             if best is None:
#                 best = match
#                 best_b = b
#             elif match.score > best.score:
#                 best = match
#                 best_b = b

#         bestMatches[a] = best_b
#     return bestMatches


# def matchInexactNames(n1, n2):
#     n1 = set(n1)
#     n2 = set(n2)

#     bestMatches12 = findBestMatches(n1, n2)
#     bestMatches21 = findBestMatches(n2, n1)

#     matches = []

#     for a in n1:
#         b = bestMatches12[a]

#         if a == bestMatches21[b]:
#             #print("%s\t%s" % (a, b))
#             matches.append((a,b))
#         else:
#             print("**"*20)
#             print("%s -> %s" % (a, bestMatches12[a]))
#             sw.align(a, bestMatches12[a]).dump()
#             print("%s -> %s" % (b, bestMatches21[b]))
#             sw.align(b, bestMatches21[b]).dump()
#             print("--")
#             print("%s -> %s" % (bestMatches12[a], bestMatches21[bestMatches12[a]] ))
#             sw.align(bestMatches12[a], bestMatches21[bestMatches12[a]]).dump()
#             print("%s -> %s" % (bestMatches21[b], bestMatches12[bestMatches21[b]] ))
#             sw.align(bestMatches21[b], bestMatches12[bestMatches21[b]]).dump()
            
#             print("**"*20)
        
#         #revMatch = sw.align( best_b, a)
#         #print("--"*20)
#         #print(a)
#         #print(best_b)
#         #best.dump()
#         #revMatch.dump()

#     return matches


# # Manually-matched identifiers
# preMatchedA = ("Bacteria_Peregrinibacteria_CG2_30_FULL_Peregrinibacteria_PER_44_17",)

# preMatchedB = ("Bacteria_CPR_Peregrinibacteria_CG_PER_02",)

# """
# Write mapping of identifiers used in Hug et al., between the tree (s6) and multiple alignment (s5).
# Many identifiers don't match identically, and in that case, we will use fuzzy text matching, based on 
# a global version of Levenshtein distance, with hand-tweaked scoring (determined using a small test-set).
# A fuzzy matching is accepted if both identifiers are each others highest scoring match.
# One taxon (see above) is matched manually because it appears truncated and has no better match.
# If this manual matching is accepted, all other taxons have good matching that pass manual inspection.
# """
# def matchTreeWithAlignment():

#     # Read the tree identifiers
#     treefmt = None
    
#     with open(nmicrobiol201648_s6, "r") as f:
#         treefmt = f.read()

#     # Strip internal support, and parse the resulting string
#     tree = PhyloTree(stripTreeInternalSupport(treefmt))

#     count = 0
#     treeNames = set()

#     for node in tree.traverse():
#         if node.is_leaf():
#             treeNames.add(node.name)
#             count += 1

#     # Read the alignment identifiers
#     alignmentNames = set()
#     for record in AlignIO.read(nmicrobiol201648_s5, "fasta"):
#         alignmentNames.add(record.id)

#     # Make all exact matches
#     exactMatches = treeNames.intersection( alignmentNames )
#     print(len(treeNames))
#     print(len(alignmentNames))
#     print(len(exactMatches))

#     # Check the manually-matched mapping
#     preMatchedNamesA = set(preMatchedA)
#     preMatchedNamesB = set(preMatchedB)
    
#     assert( len(alignmentNames.intersection(preMatchedNamesA)) == len(preMatchedNamesA) )
#     assert( len(treeNames.intersection(     preMatchedNamesB)) == len(preMatchedNamesB) )

#     # Try to fuzzily match all remaining identifiers
#     inexactMatches = matchInexactNames( alignmentNames - treeNames - preMatchedNamesA, treeNames - alignmentNames - preMatchedNamesB )
#     print("Inexact symmetrical matches: %d" % len(inexactMatches))

#     # The following identifiers are still unmatched (i.e., their best fuzzy match is not symmetrical).
#     # Note: This group should be empty
#     print("Unmatched:")
#     print(alignmentNames - exactMatches - preMatchedNamesA - set([x[0] for x in inexactMatches]))
#     print(treeNames      - exactMatches - preMatchedNamesB - set([x[1] for x in inexactMatches]))

#     # Helper generator for csv output
#     def lines():
#         for a in exactMatches:
#             assert(a in alignmentNames)
#             assert(a in treeNames)
#             yield (a, a, "exact")
#         for a,b in inexactMatches:
#             assert(a in alignmentNames)
#             assert(b in treeNames)
#             yield (a, b, "fuzzy_matching")
#         for a,b in zip(preMatchedA, preMatchedB):
#             assert(a in alignmentNames)
#             assert(b in treeNames)
#             yield (a, b, "manual")
#         for a in alignmentNames - exactMatches - preMatchedNamesA - set([x[0] for x in inexactMatches]):
#             assert(a in alignmentNames)
#             yield (a, "", "unmatched")
#         for b in treeNames      - exactMatches - preMatchedNamesB - set([x[1] for x in inexactMatches]):
#             assert(b in treeNames)
#             yield ("", b, "unmatched")

#     # Write the generator's items in CSV format
#     with open(alignmentIdentifiersToTreeIdentifiersMappingTable_csv, "w") as fout:
#         fout.write("#AlignmentId,TreeId,MatchType\n")
#         for a, b, matchType in lines():
#             fout.write("%s,%s,%s\n" % (a, b, matchType))


def convertTree():

    #tree = Phylo.read(nmicrobiol201648_s6, "newick")
    tree = Phylo.read("nmicro_s6_pruned.nw", "newick")

    leafCount = 0

    names = set()
    
    for node in tree.get_terminals():
        leafCount += 1
        #names.add(node.name)
            

    print(leafCount)
    print(len(tree.get_nonterminals()))
    print(len(names))
            

def standalone():
    import sys
    import argparse

    ret = 0

    argsParser = argparse.ArgumentParser()
    argsParser.add_argument("--prepare-translation-map", help="Create correlations table mapping Hug et. al. tree node-ids to NCBI taxon ids", action="store_true", default=False)
    argsParser.add_argument("--test-translation-map", help="Read correlations table and write extended table with BDQA fields", action="store_true", default=False)
    argsParser.add_argument("--write-inclusion-status-table", help="Create inclusion table for all nodes (for use in iTOL viewer)", action="store_true", default=False)
    argsParser.add_argument("--match-tree-with-alignment", help="Create mapping between tree node names and multiple-alignment sequences (for Hug et. al.)", action="store_true", default=False)
    argsParser.add_argument("--convert-tree", help="Convert Hug et. al. tree for use in R", action="store_true", default=False)
    args = argsParser.parse_args()

    if args.prepare_translation_map:
        ret = prepareTranslationMap()
    elif args.test_translation_map:
        ret = testTranslationMap()
    elif args.write_inclusion_status_table:
        ret = outputNodeExistenceInRnafoldDB()
    elif args.match_tree_with_alignment:
        ret = matchTreeWithAlignment()
    elif args.convert_tree:
        ret = convertTree()
    else:
        assert(False)
    
    sys.exit(ret)

if __name__=="__main__":
    standalone()
