def PatternCount(Text, Pattern):
    count = 0
    for i in range(0, len(Text) - len(Pattern)):
        if Text[i:i + len(Pattern)] == Pattern:
            count = count + 1
    return count


def FrequentWords(Text, k):
    FrequentPatterns = {}
    for i in range(0, len(Text) - k):
        pc = PatternCount(Text, Text[i:i + k])
        FrequentPatterns[pc] = Text[i:i + k]
    return max(FrequentPatterns), FrequentPatterns[max(FrequentPatterns)]


def ReverseComplement(Pattern):
    new_pat = []
    for i in range(0, len(Pattern)):
        if Pattern[i] == 'A':
            new_pat.append('T')
        elif Pattern[i] == 'T':
            new_pat.append('A')
        elif Pattern[i] == 'G':
            new_pat.append('C')
        else:
            new_pat.append('G')
    new_pat = new_pat[::-1]
    new_pat = ''.join(new_pat)
    return new_pat


def PatternMatch(Pattern, Genome):
    sp = []
    for i in range(0, (len(Genome) - len(Pattern)) + 1):
        if Pattern == Genome[i:i + len(Pattern)]:
            sp.append(str(i))
    sp = ' '.join(sp)
    return sp


def SymbolToNumber(s):
    if s == 'A':
        return 0
    elif s == 'C':
        return 1
    elif s == 'G':
        return 2
    else:
        return 3


def PatternToNumber(Pattern):
    if Pattern == '':
        return 0
    symbol = Pattern[-1]
    prefix = Pattern[:-1]
    return 4 * PatternToNumber(prefix) + SymbolToNumber(symbol)


def NumberToSymbol(n):
    if n == 0:
        return 'A'
    elif n == 1:
        return 'C'
    elif n == 2:
        return 'G'
    else:
        return 'T'


def NumberToPattern(index, k):
    if k == 1:
        return NumberToSymbol(index)
    prefixIndex = index // 4
    r = index % 4
    symbol = NumberToSymbol(r)
    PrefixPattern = NumberToPattern(prefixIndex, k - 1)
    return PrefixPattern + symbol


def ComputingFrequencies(Text, k):
    FrequencyArray = {}
    for i in range(0, (4 ** k)):
        FrequencyArray[i] = 0
    for i in range(0, len(Text) - (k - 1)):
        Pattern = Text[i:i + k]
        j = PatternToNumber(Pattern)
        FrequencyArray[j] = FrequencyArray[j] + 1
    v = FrequencyArray

    return v


def BetterClumpFinding(Genome, k, t, L):
    FrequentPatterns = []
    Clump = {}
    for i in range(0, (4 ** k) - 1):
        Clump[i] = 0
    Text = Genome[0:L]
    FrequencyArray = ComputingFrequencies(Text, k)
    for i in range(0, (4 ** k) - 1):
        if FrequencyArray[i] >= t:
            Clump[i] = 1
    for i in range(1, len(Genome) - L):
        FirstPattern = Genome[(i - 1):(i - 1) + k]
        index = PatternToNumber(FirstPattern)
        FrequencyArray[index] = FrequencyArray[index] - 1
        LastPattern = Genome[i + L - k:i + L]
        index = PatternToNumber(LastPattern)
        FrequencyArray[index] = FrequencyArray[index] + 1
        if FrequencyArray[index] >= t:
            Clump[index] = 1
    for i in range(0, (4 ** k) - 1):
        if Clump[i] == 1:
            Pattern = NumberToPattern(i, k)
            FrequentPatterns.append(Pattern)
    return FrequentPatterns


def ImmediateNeighbors(Pattern):
    Neighborhood = []
    for i in range(0, len(Pattern)):
        symbol = Pattern[i]
        for x in 'ACTG':
            if symbol != x:
                Neighbor = Pattern[:i] + x + Pattern[i + 1:]
                Neighborhood.append(Neighbor)
    return Neighborhood


def Suffix(Pattern):
    s = Pattern[1:]
    return s


def HammingDistance(s1, s2):
    h = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            h += 1
    return h


def Neighbors(Pattern, d):
    if d == 0:
        return Pattern
    if len(Pattern) == 1:
        return ['A', 'C', 'T', 'G']
    Neighborhood = []
    SuffixNeigbors = Neighbors(Suffix(Pattern), d)
    for Text in SuffixNeigbors:
        if HammingDistance(Suffix(Pattern), Text) < d:
            for x in 'ACGT':
                s = x + Text
                Neighborhood.append(s)
        else:
            s = Pattern[0] + Text
            Neighborhood.append(s)

    return Neighborhood


def ComputingFrequenciesWithMismatches(Text, k, d):
    FrequencyArray = {}
    for i in range((4 ** k)):
        FrequencyArray[i] = 0
    for i in range(len(Text) - k):
        Pattern = Text[i:i + k]

        Neighborhood = Neighbors(Pattern, d)

        for ApproximatePattern in Neighborhood:
            j = PatternToNumber(ApproximatePattern)
            FrequencyArray[j] = FrequencyArray[j] + 1

    maxvalue = max(FrequencyArray.values())
    max_list = [k for k, v in FrequencyArray.items() if v == maxvalue]
    l = []
    for n in max_list:
        l.append(NumberToPattern(n, k))
    return l
