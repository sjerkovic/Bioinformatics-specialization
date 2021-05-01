def HammingDistance(s1, s2):
    h = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            h += 1

    return h


def Compliment(Text):
    comp_text = ''
    for t in Text:
        if t == 'A':
            comp_text = comp_text + 'T'
        elif t == 'C':
            comp_text = comp_text + 'G'
        elif t == 'G':
            comp_text = comp_text + 'C'
        else:
            comp_text = comp_text + 'A'
    return comp_text[::-1]


def APMP(Pattern, Text, d):
    s = ''
    count = 0
    Pattern_c = Compliment(Pattern)
    for i in range(0, (len(Text) - len(Pattern)) + 1):
        Pattern2 = Text[i:i + len(Pattern)]
        h1 = HammingDistance(Pattern, Pattern2)
        h2 = HammingDistance(Pattern_c, Pattern2)
        if h1 <= d:
            s = s + str(i) + ' '
            count += 1
        if h2 <= d:
            count += 1
    return count


def FrequentWords2(Text, k, d):
    p = product('ACGT', repeat=k)
    Patterns = []
    for i in p:
        s = ''.join(i)
        Patterns.append(s)
    pattern_dict = {}
    for pattern in Patterns:
        c = APMP(pattern, Text, d)
        pattern_dict[pattern] = c
    maxval = max(pattern_dict.values())
    max_list = [k for k, v in pattern_dict.items() if v == maxval]
    return ' '.join(max_list), maxval
