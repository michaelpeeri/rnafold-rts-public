# Compress and de-compress nucleic acid sequences using a Huffman code
#
from __future__ import print_function
from builtins import hex
from builtins import str
from builtins import map
from builtins import range
from builtins import object
import struct
import itertools
import random
from datetime import datetime
#
# Pre-computed Huffman code, computed for alphabet=frozenset(('a','c','g','t','n')), weights={'a':0.249,'c':0.249,'g':0.249,'t':0.249,'n':0.004}
# Uridine ('u') and uncertainty symbols other than 'n' are not supported.
# The somewhat arbitrary probabilities reflect an assumption that 'n' symbols are very rare
# 
# Notes:
# 1) This code does not treat all bases fairly: one base (in this case 't') is encoded using 3 bits instead of 2.
# 2) Based on the assumed probabilities, the expected average encoded length is 2.3 bits/nt, which is a saving 1:3.5 (or 71%) compared
#    to 8-bit text encoding. This length can vary between ~2.1-2.4 bits/nt depending on the GC content. The worst case (poly-T) is 3 bits/nt.
#
nucleotideHuffmanCode = {'a':'11', 'c':'10', 'g':'00', 't':'011', 'n':'010'}
reverseNucHuffmanCode = {'11':'a', '10':'c', '00':'g', '011':'t', '010':'n'}


class Bytestream(object):
    bitStringToInt = {'0':0b0, '1':0b1}

    def __init__(self, initialData = None):
        if( not initialData is None):
            self._b = bytearray(initialData)
        else:
            self._b = bytearray()

        self._bytePos = 0
        self._bitPos = 0
        if( initialData is None ):
            self._b.insert(0, 0) # insert the first bytes

    def push(self, bits):
        #print("Adding %s" % bits)
        #for b in reversed(bits):
        for b in bits:
            bit = Bytestream.bitStringToInt[b]
            
            if( self._bitPos == 8 ):
                # Add an extra byte
                self._bytePos += 1
                self._b.insert(self._bytePos, 0)
                self._bitPos = 0
            
            if( bit == 1):
                mask = bit << (self._bitPos)
                self._b[self._bytePos] |= mask

            self._bitPos += 1

    def skipBits(self, numBits):
        # TODO: Replace with O(1) impl.
        for i in range(numBits):
            self.push('0')

    def countBits(self):
        return( self._bytePos*8 + self._bitPos )

    def getNextBit(self):
        if( self._bitPos == 8 ):
            # Add an extra byte
            self._bytePos += 1
            self._bitPos = 0

        # Get the next byte
        byte = self._b[self._bytePos]

        # Read the correct bit
        mask = 1<<self._bitPos
        bit = int((mask & byte) != 0)
        #print(bin(byte), bin(mask), bin(bit))

        assert(bit==0b1 or bit==0b0)

        # Move to the next pos
        self._bitPos += 1

        return bit



def encode(seq):
    seq = seq.lower()
    n = len(seq)

    out = Bytestream()

    #encode length
    # TODO - implement this using struct...
    out.push(reversed(str(bin(n))[2:]))
    out.skipBits( 32 - out.countBits())
    assert(out.countBits()==32)

    for i, symbol in enumerate(seq):
        codeword = nucleotideHuffmanCode[symbol]
        out.push(codeword)

    #print((out.countBits()-32)/8, n*2.3/8)

    return out._b


def decode(data):
    header = data[:4]
    content = data[4:]

    n = struct.unpack('I', header)[0]
    #print("Decoding %d symbols..." % (n,))
 
    stream = Bytestream(content)

    decoded = []

    word = []
    while(True):
        word.append( stream.getNextBit() )
        x = "".join(map(str,word))
        #print("Trying %s (%s)" % (x, word))
        if( x in reverseNucHuffmanCode ):
            #print("%s -> %s" % (x, reverseNucHuffmanCode[x]) )
            decoded.append( reverseNucHuffmanCode[x] )
            word = []

        if(len(decoded)==n):
            break
    
    return "".join(decoded)


def test(seq, debug=True):
    uncompressedSizeBytes = len(seq)
    seq = seq.lower()
    encoded = encode(seq)

    compressedSizeBytes = len(encoded)

    decoded = decode(encoded)

    if( decoded != seq ):
        print("------------------------------")
        print(seq)
        print("Failed!")
        assert(False)

    if( debug):
        print("Length: %dnt Compressed size %.2g%%" % ( uncompressedSizeBytes, float(compressedSizeBytes)/uncompressedSizeBytes*100))


"""
Generate a random sequence of length N
"""
def randseq(N, alphabet='acgt'):
    return ''.join([c for c in itertools.starmap(lambda:random.choice(alphabet), itertools.repeat((), N))])

"""
Exhaustively generate all sequences of length N
"""
def allseq(N):
    if( N==1 ):
        yield "a"
        yield "c"
        yield "g"
        yield "t"
        yield "n"
    else:
        for rest in allseq(N-1):
            for j in ('a','c','g','t','n'):
                yield j+rest


def test_all():
    a = Bytestream()
    a.push('011')
    #a.push('10')
    #a.push('01')
    print(repr(a._b))
    print(hex(0b00000110))

    a.push('10')
    print(repr(a._b))
    print(hex(0b00010110))

    a.push('0')
    print(repr(a._b))
    print(hex(0b00010110))

    a.push('1')
    print(repr(a._b))
    print(hex(0b01010110))
    
    a.push('1')
    print(repr(a._b))
    print(hex(0b11010110))



#for testval in range(10000):
#    a = bytestream()
#    a.push(str(bin(testval))[2:])

#    print(bin(testval), hex(testval), repr(a._b), a.countBits())


    test("t")
    test("a")
    test("g")
    test("c")
    test("n")
    test("at")
    test("atc")


    test("atcgacgacgatctaCgatcgAAAggtctagcta")
    test("atcgacgacgatctacgatcgaaaggtctagctat")
    test("atcgacgacgatctacgatcgaaaggtctagctatg")
    test("atcgacgacgatctnacgatcgaaaggtctagctatgc")
    test("atcgacgacgatctnnacgatcgaaaggtctagctatgc")
    test("atcgacgacgatctnnnacgatcgaaaggtctagctatgc")
    test("atcgacgacgatctnnnncgatcgaaaggtctagctatgc")
    test("atcgacgacgatctnnnnngatcgaaaggtctagctatgc")
    test("atcgacgacgatctnnnnnnatcgaaaggtctagctatgc")
    test("atcgacgacgatctnnnnnnntcgaaaggtctagctatgc")
    test("atcgacgacgatctnnnnnnntcgaaaggtctagctatgcn")
    test("atcgacgacgatctnnnnnnntcgaaaggtctagctatgcnn")
    test("atcgacgacgatctnnnnnnntcgaaaggtctagctatgcnnn")



    test("a"*100)
    test("c"*100)
    test("g"*100)
    test("t"*100)
    test("n"*100)
    test("acgt"*25)

    # Test on random sequences of varying length
    for i in range(1, 100):
        for j in range(100):
            test(randseq(i))

    # Test on some random sequences
    for i in range(1000):
        test(randseq(5000))

    # Exhaustively test all sequences of length N (1>= N >= 11)
    for n in range(1,12):
        print("-"*50)
        total = 5**n
        print("%s - Testing %d sequences of length %d" % (datetime.now().isoformat(), total, n))
        count = 0
        for i in allseq(n):
            test(i, False)
            count += 1
            if( count%20000==19999 ):
                print("%s - %.2g%% done" % (datetime.now().isoformat(), float(count)/total*100))


    return 0

        

if __name__=="__main__":
    import sys
    sys.exit(test_all())






