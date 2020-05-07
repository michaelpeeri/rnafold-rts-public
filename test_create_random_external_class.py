from data_helpers import SpeciesCDSSource
from random import shuffle
import argparse


def getAllProteins(taxId):
    return list( SpeciesCDSSource(taxId) )

        


argsParser = argparse.ArgumentParser()
argsParser.add_argument( "--taxid", type=int )
argsParser.add_argument( "--num",   type=int, default=700 )
args = argsParser.parse_args()


allProtIds = getAllProteins(args.taxid)
shuffle( allProtIds )
shuffle( allProtIds )
shuffle( allProtIds )
for protId in allProtIds[:args.num]:
    print(protId)
