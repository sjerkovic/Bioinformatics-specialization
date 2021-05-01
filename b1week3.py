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


def HammingDistance(s1, s2):
    h = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            h += 1

    return h


def Suffix(Pattern):
    s = Pattern[1:]
    return s


def motifenumeration(Dna, k, d):
    patterns = set()
    # Checks each seqeunce
    for seq in Dna:
        # Checks each kmer in a sequence
        for i in range(len(seq) - k):
            pattern = seq[i:i + k]
            # Checks kmer against each kmer in a sequence for mismatches
            for j in range(len(seq) - k):
                test = seq[j:j + k]
                mismatches = findMismatches(pattern, test, d)
                if mismatches <= d:
                    if sharedMotif(Dna, test, d) == True:
                        patterns.add(test)
    return patterns


# Finds mismatches in similar length strings
def findMismatches(kmer, test, d):
    mismatches = 0
    for i in range(len(kmer)):
        if mismatches <= d:
            if kmer[i] != test[i]:
                mismatches += 1
        else:
            break
    return mismatches


# Checks if possible motif is in every dna string
def sharedMotif(Dna, kmer, d):
    # for loop to go through get seq in dna
    ##loop thorugh and see if kmer is in each string
    ##return true if so then add kmer to the set
    shared = False
    shares = 0
    for seq in Dna:
        found = False
        for i in range(len(seq) - len(kmer)):
            test = seq[i:i + len(kmer)]
            if findMismatches(kmer, test, d) <= d:
                print(kmer + " " + test)
                found = True
                shares += 1
                break
        if found == False:
            break
    if shares == len(Dna):
        shared = True
    return shared


def Pmpk(Text, k, Profile):
    pattern_dict = {}
    for i in range(len(Text) - k):
        Pattern = Text[i:i + k]
        pp = Probability(Profile, Pattern)
        pattern_dict[Pattern] = pp
    maxval = max(pattern_dict.values())
    max_pat = [k for k, v in pattern_dict.items() if v == maxval]
    return max_pat


def FormProfile(Motif):
    Profile = pd.DataFrame(columns=range(0, len(Motif[0])), index=['A', 'C', 'G', 'T'])
    for i in range(len(Motif[0])):
        a_count = 0
        c_count = 0
        g_count = 0
        t_count = 0
        for j in Motif:
            if j[i] == 'A':
                a_count += 1
            elif j[i] == 'C':
                c_count += 1
            elif j[i] == 'G':
                g_count += 1
            elif j[i] == 'T':
                t_count += 1
            Profile[i] = [a_count / len(Motif), c_count / len(Motif), g_count / len(Motif), t_count / len(Motif)]
    return Profile


def Score(Motif):
    score = 0
    for i in range(len(Motif[0])):
        a_count = 0
        c_count = 0
        g_count = 0
        t_count = 0
        for j in range(len(Motif)):

            if Motif[j][i] == 'A':
                a_count += 1
            elif Motif[j][i] == 'C':
                c_count += 1
            elif Motif[j][i] == 'G':
                g_count += 1
            elif Motif[j][i] == 'T':
                t_count += 1
        max_count = max(a_count, c_count, g_count, t_count)
        sum_count = a_count + c_count + g_count + t_count
        score = score + (sum_count - max_count)
    return score


def GreedyMotifSearch(Dna, k, t):
    BestMotifs = []
    best_score = math.inf
    # for s in Dna:
    # BestMotifs.append(s[:k])
    for i in range(len(Dna[0]) - k):
        Motif = [Dna[0][i:i + k]]
        for j in range(1, t):
            string = Dna[j]
            Profile = FormProfile(Motif)
            Motif.extend(Pmpk(string, k, Profile))
        if Score(Motif) < best_score:
            best_score = Score(Motif)
            BestMotifs = Motif
    return BestMotifs
