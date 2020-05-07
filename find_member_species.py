# Utility script to list species that belong to a specific parent taxon
import random
import argparse
from ncbi_taxa import ncbiTaxa
from data_helpers import allSpeciesSource

def parseList(conversion=str):
    def convert(values):
        return map(conversion, values.split(","))
    return convert

argsParser = argparse.ArgumentParser()
argsParser.add_argument("--from-species", type=parseList(int), default=())
argsParser.add_argument("--taxon", type=int)

args = argsParser.parse_args()

if not args.from_species: # not species listed; use all species
    fromSpecies = frozenset(allSpeciesSource())
else: # use only species listed
    fromSpecies = args.from_species


positiveResults = []
negativeResults = []

for taxid in fromSpecies:

    lineage = ncbiTaxa.get_lineage( taxid )
    if args.taxon in lineage:
        positiveResults.append(taxid)
    else:
        negativeResults.append(taxid)
        
print("Positive results (N={}):".format( len(positiveResults)) )
print(",".join(map(str,positiveResults)))
print("Negative results (N={}):".format( len(negativeResults)) )
print(",".join(map(str,negativeResults)))
