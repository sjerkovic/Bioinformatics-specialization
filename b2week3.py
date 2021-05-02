import itertools


def ProteinTranslation(Pattern, GeneticCode):
    string = ''
    for i in range(0, len(Pattern) - 3, 3):
        kmer = Pattern[i:i + 3]
        aa = GeneticCode[kmer]
        string = string + aa
    return string


def Transcription(String):
    new_string = ''
    for i in String:
        if i == 'U':
            new_string += 'T'
        else:
            new_string += i
    return new_string


def ReverseCompliment(String):
    new_string = ''
    for i in String:
        if i == 'A':
            new_string += 'T'
        elif i == 'C':
            new_string += 'G'
        elif i == 'G':
            new_string += 'C'
        elif i == 'T':
            new_string += 'A'
    return new_string[::-1]


def PeptideToCode(String, GeneticCode):
    reverse_aa_dict = {}
    for k, v in GeneticCode.items():
        reverse_aa_dict[v] = reverse_aa_dict.get(v, [])
        reverse_aa_dict[v].append(k)
    code = []
    for i in String:
        code.append(reverse_aa_dict[i])

    l = list(itertools.product(*code))
    result = []
    for i in l:
        result.append(''.join(i))
    return result


def RabinKarpSet(string, strings):
    hsubs = []
    patterns = []
    m = len(strings[0])
    n = len(string)
    for s in strings:
        hsubs.append(hash(s))
    hs = hash(string[0:m])
    for i in range(n - m + 1):
        if hs in hsubs and string[i:i + m] in strings:
            patterns.append(string[i:i + m])
        hs = hash(string[i + 1:i + m + 1])
    return patterns


def PeptideEncoding(Text, Peptide, GeneticCode):
    # all_threemers = []
    # for i in range((len(Text)-len(Peptide)*3)+1):
    # all_threemers.append(Text[i:i+len(Peptide)*3])
    initial_code = PeptideToCode(Peptide, GeneticCode)
    code = []
    for i in initial_code:
        code.append(Transcription(i))
    final_code = []
    for i in code:
        final_code.append(i)
        final_code.append(ReverseCompliment(i))
    # result = []
    # for i in all_threemers:
    # for j in final_code:
    # if i==j:
    # result.append(i)
    result = RabinKarpSet(Text, final_code)
    # for i in result:
    # print(i)
    return result


# For cyclic peptides:
def FindSubpeptides(Peptide):
    all_strings = []
    for i in range(len(Peptide)):
        all_strings.append(Peptide[i:] + Peptide[:i])
    all_subpeptides = []
    for i in range(len(all_strings[0])):
        for j in all_strings:
            sp = j[0:i]
            all_subpeptides.append(sp)
    all_subpeptides[:] = [i for i in all_subpeptides if i != '']
    return all_subpeptides


def FindLinearSubpeptides(Peptide):
    subpeptides = []
    for i in range(len(Peptide)):
        for j in range(len(Peptide), i, -1):
            subpeptides.append(Peptide[i:j])
    return subpeptides


def TheoreticalSpectrum(Peptide):
    # sp = FindSubpeptides(Peptide) - for cyclic peptide
    # for linear peptide:
    sp = FindLinearSubpeptides(Peptide)
    masses = []
    for i in sp:
        mass = 0
        for j in i:
            mass += mass_dict[j]
        masses.append(mass)
    masses.append(0)
    total_mass = 0
    for i in Peptide:
        total_mass += mass_dict[i]
    # masses.append(total_mass) - for cyclic peptide
    masses.sort()
    # for m in masses:
    # print(m)
    return masses


def LinearSpectrum(Peptide, Alphabet, AminoAcidMass):
    PrefixMass = {}
    PrefixMass[0] = 0
    for i in range(1, len(Peptide) + 1):
        for s in Alphabet:
            if s == Peptide[i - 1]:
                PrefixMass[i] = PrefixMass[i - 1] + AminoAcidMass[s]
    LinearSpectrum = [0]
    print(PrefixMass)
    for i in range(len(Peptide)):
        for j in range(i + 1, len(Peptide) + 1):
            LinearSpectrum.append(PrefixMass[j] - PrefixMass[i])
    LinearSpectrum = sorted(LinearSpectrum)
    return LinearSpectrum


def LinearSpectrumWithMasses(Peptide):
    PrefixMass = {}
    PrefixMass[0] = 0
    for i in range(1, len(Peptide) + 1):
        PrefixMass[i] = PrefixMass[i - 1] + Peptide[i - 1]
    LinearSpectrum = [0]

    for i in range(len(Peptide)):
        for j in range(i + 1, len(Peptide) + 1):
            LinearSpectrum.append(PrefixMass[j] - PrefixMass[i])
    LinearSpectrum = sorted(LinearSpectrum)
    return LinearSpectrum


def CyclicSpectrum(Peptide, Alphabet, AminoAcidMass):
    PrefixMass = {}
    PrefixMass[0] = 0
    for i in range(1, len(Peptide) + 1):
        for s in Alphabet:
            if s == Peptide[i - 1]:
                PrefixMass[i] = PrefixMass[i - 1] + AminoAcidMass[s]
    peptideMass = PrefixMass[len(Peptide)]
    CyclicSpectrum = [0]
    for i in range(len(Peptide)):
        for j in range(i + 1, len(Peptide) + 1):
            CyclicSpectrum.append(PrefixMass[j] - PrefixMass[i])
            if i > 0 and j < len(Peptide):
                CyclicSpectrum.append(peptideMass - (PrefixMass[j] - PrefixMass[i]))
    CyclicSpectrum = sorted(CyclicSpectrum)
    return CyclicSpectrum


def CyclicSpectrumWithMasses(Peptide):
    PrefixMass = {}
    PrefixMass[0] = 0
    for i in range(1, len(Peptide) + 1):
        PrefixMass[i] = PrefixMass[i - 1] + Peptide[i - 1]
    peptideMass = PrefixMass[len(Peptide)]
    CyclicSpectrum = [0]
    for i in range(len(Peptide)):
        for j in range(i + 1, len(Peptide) + 1):
            CyclicSpectrum.append(PrefixMass[j] - PrefixMass[i])
            if i > 0 and j < len(Peptide):
                CyclicSpectrum.append(peptideMass - (PrefixMass[j] - PrefixMass[i]))
    CyclicSpectrum = sorted(CyclicSpectrum)
    return CyclicSpectrum


def SubpeptidesOfAPeptide(n):
    s = 0
    for i in range(n, 1, -1):
        s += i
    return s + 2


def Expand(Peptides, Spectrum):
    # masses = [57,71,87,97,99,101, 103, 113, 114, 115, 128, 129, 131, 137, 147, 156, 163, 186]

    z = []
    for p in Peptides:
        for i in range(len(Spectrum)):
            c = Spectrum[i]
            n = p[:]
            n.append(c)
            z.append(n)
    return z


def Consistent(Peptide, Spectrum):
    ls = LinearSpectrumWithMasses(Peptide)
    s = Spectrum[:]
    for i in ls:
        if i in s:
            s.remove(i)
        else:
            return False
    return True


def CycloSpectrum(Peptide):
    spectrum = []
    spectrum.extend(Peptide)
    spectrum.extend([x + y for x, y in itertools.combinations(Peptide, 2)])
    spectrum.append(0)
    spectrum.append(sum(Peptide))
    return sorted(spectrum)


def CyclopeptideSequencing(Spectrum):
    Spectrum = Spectrum.split()
    Spectrum = [int(v) for v in Spectrum]
    # just_subpeptides = Spectrum[1:4]
    masses = [57, 71, 87, 97, 99, 101, 103, 113, 114, 115, 128, 129, 131, 137, 147, 156, 163, 186]
    CandidatePeptides = [[v] for v in Spectrum]

    FinalPeptides = []
    ParentMass = Spectrum[-1]
    while len(CandidatePeptides) > 0:

        CandidatePeptides = Expand(CandidatePeptides, masses)
        f = CandidatePeptides[:]
        # print(CandidatePeptides[0])
        # print('\n')
        for Peptide in f:

            p_index = CandidatePeptides.index(Peptide)
            if sum(Peptide) == ParentMass:
                if CyclicSpectrumWithMasses(Peptide) == Spectrum and Peptide not in FinalPeptides:
                    FinalPeptides.append(Peptide)
                del CandidatePeptides[p_index]
            elif Consistent(Peptide, Spectrum) == False:
                del CandidatePeptides[p_index]

    for i in FinalPeptides:
        print(*i, sep='-')


def Trim(Leaderboard, Spectrum, N):
    LinearScores = {}
    for j in range(len(Leaderboard)):
        Peptide = Leaderboard[j]
        LinearScores[j] = Score(Peptide, Spectrum)
    lin_scores_values = [v for k, v in LinearScores.items()]
    Leaderboard = [x for _, x in sorted(zip(lin_scores_values, Leaderboard), reverse=True)]
    LinearScores = {k: v for k, v in sorted(LinearScores.items(), key=lambda item: item[1], reverse=True)}
    for j in range(N, len(Leaderboard)):
        if LinearScores[j - 1] < LinearScores[N - 1]:
            del Leaderboard[j:]
            return Leaderboard
    return Leaderboard


def LeaderboardCyclopeptideSequencing(Spectrum, N):
    Spectrum = Spectrum.split()
    Spectrum = [int(v) for v in Spectrum]
    Leaderboard = [[v] for v in Spectrum]
    masses = [57, 71, 87, 97, 99, 101, 103, 113, 114, 115, 128, 129, 131, 137, 147, 156, 163, 186]

    LeaderPeptide = ''
    ParentMass = Spectrum[-1]
    while len(Leaderboard) != 0:
        Leaderboard = Expand(Leaderboard, masses)
        for Peptide in Leaderboard:
            if sum(Peptide) == ParentMass:
                if Score(Peptide, Spectrum) > Score(LeaderPeptide, Spectrum):
                    LeaderPeptide = Peptide
            elif sum(Peptide) > ParentMass:
                Leaderboard.remove(Peptide)
        Leaderboard = Trim(Leaderboard, Spectrum, N)
    return Leaderboard
