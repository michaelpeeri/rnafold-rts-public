import re
import subprocess

# ----------------------------------------------------------
# configuration
vienna_rnafold_path = "~/anaconda2N/bin/RNAfold"
#vienna_rnafold_path = "~/anaconda2/bin/RNAfold"
# ----------------------------------------------------------


# Parse vienna 'rnafold' output
# Example:
#
#
#[line 1]:   ccagucgaccagacuauauacaaccuacgcguaucgcgcga
#[line 2]:   ..((((.....))))............(((((...))))). ( -9.50)
#
reMFEScore = re.compile(".*\n.*[(]\s*([\d.-]+)[)]\n")   # parse the score (group 1), ignore the structure
reMFEScoreWithStructure = re.compile(".*\n([().]+)\s+[(]\s*([\d.-]+)[)]\n")  # return the score (group 2) and structure (group 1)

def RNAfold_direct(seq:str, explicitCalculationTemperature:float =None) -> float:
    
    if explicitCalculationTemperature is None:
        cmdline = "echo %s | %s --noPS" % (seq, vienna_rnafold_path)
    else:
        cmdline = "echo %s | %s --temp=%g --noPS" % (seq, vienna_rnafold_path, explicitCalculationTemperature)
        
    out = str( subprocess.check_output(cmdline, shell=True), encoding="ascii" )
    score = float(reMFEScore.match(out).group(1))
    assert(score<=0.0)
    return score


"""
Parse the 2nd-ary structure string representation returned by vienna,
e.g.: (((.....)))..........((.(((((...))))).)).'

return pairs of indices of paired (hybridized) nucleotides.

if cdsOffset is given, all coordinates will be offset by that length (used to convert coordinates from
window-referenced to CDS-referenced, so structures from different windows can be combined).
"""
def parseStructure(struct, cdsOffset=0):
    stack = []

    ret = []

    for pos, s in enumerate(struct):
        if s=='(':
            stack.append(pos)
        elif s==')':
            opposingPos = stack.pop()
            assert(not opposingPos is None)
            assert(opposingPos < pos)  # this will obviously always be true...
            #print("%d-%d" % (opposingPos, pos))
            ret.append( (opposingPos+cdsOffset, pos+cdsOffset) )
        elif s=='.':
            pass
        else:
            raise Exception("Structure representation contains unsupported character '%s'" % s)
    assert(len(stack)==0)  # after parsing, all pairs must have been matched...

    return sorted(ret)
    
    

def RNAfoldWithStructure(seq, cdsOffset=0):
    out = subprocess.check_output("echo %s | %s --noPS" % (seq, vienna_rnafold_path), shell=True)
    match = reMFEScoreWithStructure.match(out)
    score = float(match.group(2))
    assert(score<=0.0)

    struct = match.group(1)
    parsedPairs = parseStructure(struct, cdsOffset)

    return (parsedPairs, score)
    

def testsMain():
    print(RNAfoldWithStructure("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
    print(RNAfoldWithStructure("gucgaccagacuauauacaaccuacgcguaucgcgcgaagc"))
    print(RNAfoldWithStructure("cgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgcgc"))
    print(RNAfoldWithStructure("cgcgcgcgcgcgcgcgcgcgcatatatatatatatatatat"))
    print(RNAfoldWithStructure("cgcgcgcgctatatatatatatatatatatatagcgcgcgc"))
    print(RNAfoldWithStructure("atatatatacgcgcgcgcgcgcgcgcgcgcgcgtatatata"))
    return 0
    
if __name__=="__main__":
    import sys
    sys.exit(testsMain())
    
    
