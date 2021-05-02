import collections


def StringComposition(k, Text):
    kmers = []
    for i in range((len(Text) - k) + 1):
        kmer = Text[i:i + k]
        if kmer not in kmers:
            kmers.append(kmer)
    return sorted(kmers)


def StringReconstruction(Strings):
    end = ''
    for i in range(1, len(Strings)):
        last_letter = Strings[i][-1]
        end = end + last_letter
    return print(Strings[0] + end)


def OverlapGraph(Patterns):
    pattern_dict = {}
    for Pattern1 in Patterns:
        for Pattern2 in Patterns:
            if Pattern1 == Pattern2:
                pass
            else:
                if Pattern1[1:] == Pattern2[:-1]:
                    if Pattern1 in pattern_dict:
                        pattern_dict[Pattern1].append(Pattern2)
                    else:
                        pattern_dict[Pattern1] = [Pattern2]
    for k, v in pattern_dict.items():
        print(k + ' -> ' + ','.join(v))


def generate_binary_kmers(k):
    for i in range(2 ** k):
        res = bin(i).lstrip('0b')
        yield '0' * (k - len(res)) + res


def check_universal(test_string, k):
    for kmer in generate_binary_kmers(k):
        if kmer not in test_string:
            return False
    return True


def find_universal_string(k):
    optimal_string_length = (2 ** k) + (k - 1)
    for test_string in generate_binary_kmers(optimal_string_length):
        if check_universal(test_string, k):
            return (test_string)


def DeBrujin(k, Text):
    pattern_dict = {}
    for i in range(len(Text) - k + 1):
        Pattern = Text[i:(i + k) - 1]
        if Pattern not in pattern_dict:
            pattern_dict[Pattern] = [Text[i + 1:(i + 1 + k) - 1]]
        else:
            pattern_dict[Pattern].append(Text[i + 1:(i + 1 + k) - 1])
    for v in pattern_dict.values():
        v = v.sort()
    pattern_dict = collections.OrderedDict(sorted(pattern_dict.items()))
    for k, v in pattern_dict.items():
        print(k + ' -> ' + ','.join(v))


def DeBrujinFromKmers(Patterns):
    pattern_dict = {}
    for Pattern in Patterns:
        if Pattern[:-1] not in pattern_dict:
            pattern_dict[Pattern[:-1]] = [Pattern[1:]]
        else:
            pattern_dict[Pattern[:-1]].append(Pattern[1:])
    for v in pattern_dict.values():
        v = v.sort()
    pattern_dict = collections.OrderedDict(sorted(pattern_dict.items()))
    for k, v in pattern_dict.items():
        print(k + ' -> ' + ','.join(v))
