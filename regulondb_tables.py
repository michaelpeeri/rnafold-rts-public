import csv
import re



f1 = "./data/regulondb/UTR_5_3_sequence.txt"
id_conversion_file = "/tamir1/mich1/termfold/data/Ensembl/Ecoli/identifiers.tsv"

def getIdentifiersMapping():
    ret = {}
    with open(id_conversion_file, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            assert(len(row)==3)
            ret[row[1]] = row[0]
            ret[row[2]] = row[0]
    return ret

reGeneNameAndPosition = re.compile("""(['\w-]+)[(][^)]+[)]""")
reGeneCoords =          re.compile("""(\S+)[(](\d+)[,](\d+)[)]""")
reUTRCoords  =          re.compile("""(\d+)-(\d+)""")
def readUTRsTable():
    ret = {}
    numSkipped = 0 
    with open(f1) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if len(row)==1 and row[0][0]=="#": continue
            lastGeneName = reGeneNameAndPosition.match( row[6] ).group(1)
            assert( lastGeneName )

            firstGeneInfo = reGeneCoords.match( row[5] ).groups()
            lastGeneInfo  = reGeneCoords.match( row[6] ).groups()
            UTRinfo       = reUTRCoords.match(  row[8] ).groups()

            #if len(row)<12: # No info on 3'UTR
            #    numSkipped += 1
            #    continue

            utr5primeLength = int(firstGeneInfo[1]) - int(UTRinfo[0])
            utr3primeLength = int(UTRinfo[1]) - int(lastGeneInfo[2])

            #if utr3primeLength > 50:
            #    print("Z")
            #else:
            #    print("W")
            if utr3primeLength < 30 and utr3primeLength>0 :
                print("xxt")
            
            if row[7]:
                terminatorInfo = reGeneCoords.match( row[7] ).groups()
            else:
                #if utr3primeLength > 0:
                #    print("Y")
                #print("X {}".format( utr3primeLength ))
                #continue # skip genes missing terminator info
                terminatorInfo = None


            #if utr3primeLength>1000:
            #    print("__")
                
            #ThreePrimeUtrCoords = tuple(map(int, row[11].split('-')))
            #ThreePrimeUtrCoordsLen = ThreePrimeUtrCoords[1]-ThreePrimeUtrCoords[0]
            #assert(ThreePrimeUtrCoordsLen>0)

            print("{}--[##########]--{} {}".format(utr5primeLength, utr3primeLength, terminatorInfo))

            ret[lastGeneName] = utr3primeLength
            
    print("Skipped: {}".format(numSkipped))
    return ret

def get3primeUtrLength(taxId:int) -> map:
    if taxId==511145:
        return readUTRsTable()
    else:
        return {}

if __name__=="__main__":
    import sys
    from mfe_plots import saveHistogram
    import numpy as np
    data = readUTRsTable()
    saveHistogram( np.array(list(data.values())), "regulondb_3utr_lengths.pdf", bins=(0,15,25,50,75,100,150,200,1000,2000,4000) )
    
    sys.exit(0)
