import collections


def graph_to_dict(Graph):
    s = Graph.split('\n')
    s = [v.split(' -> ') for v in s]
    k = [v[0] for v in s]
    v = [v[1].split(',') for v in s]
    d = {}
    for i in range(len(k)):
        d[k[i]] = v[i]
    return d


def dict_to_edges(d):
    edges = []
    for k, v in d.items():
        for val in v:
            edges.append((k, val))
    return edges


def Euclidian(e):
    edges = e[:]
    nodes = [v[0] for v in edges]
    n1 = nodes[0]
    stack = []
    stack.append(n1)
    circuit = []
    while len(edges) > 0:
        if n1 in nodes:
            for edge in edges:
                if edge[0] == n1:
                    nodes.remove(n1)
                    stack.append(edge[1])
                    n1 = edge[1]


        else:
            n1 = stack[-1]
            circuit.append(stack[-1])
            stack.remove(stack[-1])

    return circuit


def reorder(l, x):
    new_l = []
    new_l.extend(l[l.index(x):])
    new_l.extend(l[0:l.index(x)])
    return new_l


def CycleToPath(Graph):
    e = dict_to_edges(graph_to_dict(Graph))
    # e = Graph
    # for k,v in Graph.items():
    # e.append((k,v))

    n = [v[0] for v in e]
    an = [v[1] for v in e]
    x = []

    for i in set(n + an):
        if n.count(i) != an.count(i):
            x.append(i)
    if n.count(x[0]) > an.count(x[0]):
        tupl = tuple(x[::-1])

    else:
        tupl = tuple(x)

    e.append(tupl)
    nodes = [v[0] for v in e]
    adj_nodes = [w[1] for w in e]
    edges = e[:]
    stack = [nodes[0]]
    circuit = []
    n = nodes[0]
    # stack.append(n)

    while len(circuit) != len(e):
        if n in [u[0] for u in edges]:
            n_index = nodes.index(n)
            adj_node = adj_nodes[n_index]
            tup = (n, adj_node)
            stack.append(adj_node)

            # edges.remove(tup)
            del edges[edges.index(tup)]
            # nodes.remove(n)
            del nodes[n_index]
            # adj_nodes.remove(adj_node)
            del adj_nodes[n_index]
            n = adj_node

        else:

            circuit.append(stack[-1])
            stack.pop(-1)

            n = stack[-1]

    r = circuit[::-1]
    r = reorder(r, tupl[1])
    result = ''
    for i in r:
        result = result + str(i) + '->'
    result = result[:-2]
    return r


def DeBrujinFromKmers(Patterns):
    Patterns = Patterns.split('\n')
    pattern_dict = {}
    for Pattern in Patterns:
        if Pattern[:-1] not in pattern_dict:
            pattern_dict[Pattern[:-1]] = [Pattern[1:]]
        else:
            pattern_dict[Pattern[:-1]].append(Pattern[1:])
    for v in pattern_dict.values():
        v = v.sort()
    pattern_dict = collections.OrderedDict(sorted(pattern_dict.items()))
    r = ''
    for k, v in pattern_dict.items():
        r += (k + ' -> ' + ','.join(v)) + '\n'
    r = r[:-1]
    return r


def GenomeReconstruction(Patterns):
    dB = DeBrujinFromKmers(Patterns)
    path = CycleToPath(dB)
    Text = StringReconstruction(path)
    return Text


def CircularString(k):
    l = []
    for i in itertools.product('10', repeat=k):
        l.append(i)
    kmers = [''.join(i) for i in l]
    s = ''
    for i in kmers:
        s += i + '\n'
    s = s[:-1]
    d = DeBrujinFromKmers(s)
    e = Euler_new(d)
    return StringReconstruction(e)[:-(k - 1)]


def Kdmer(k, d, Text):
    kdmers = []
    for i in range(len(Text) - (2 * k + d - 1)):
        Pattern1 = Text[i:i + k]
        Pattern2 = Text[i + k + d:i + 2 * k + d]
        tup = (Pattern1, Pattern2)
        kdmers.append(tup)
    js = sorted([''.join(v) for v in kdmers])
    r = ''
    for i in js:
        r += '({p1}|{p2}) '.format(p1=i[:k], p2=i[k:])
    return r


def SSBGP(GappedPatterns, k, d):
    # GappedPatterns=GappedPatterns.split('\n')
    FirstPatterns = [v[0] for v in GappedPatterns]
    SecondPatterns = [v[1] for v in GappedPatterns]
    PrefixString = StringReconstruction(FirstPatterns)
    SuffixString = StringReconstruction(SecondPatterns)
    for i in range(k + d + 1, len(PrefixString)):
        if PrefixString[i] != SuffixString[i - k - d]:
            return print('there is no string spelled by the gapped path')
        else:
            return PrefixString + SuffixString[-(k + d):]


def DeBrujinFromPairs(Pairs):
    Pairs = Pairs.split('\n')
    pattern_dict = {}
    for Pattern in Pairs:
        Prefix0 = Pattern.split('|')[0][:-1]
        Prefix1 = Pattern.split('|')[1][:-1]
        Suffix0 = Pattern.split('|')[0][1:]
        Suffix1 = Pattern.split('|')[1][1:]
        Node_p = (Prefix0, Prefix1)
        Node_s = (Suffix0, Suffix1)
        if Node_p not in pattern_dict:
            pattern_dict[Node_p] = Node_s
        else:
            pattern_dict[Node_p].append(Node_s)

    # r = ''

    # for k,v in pattern_dict.items():
    # r+=(str(k)+' -> '+str(v))+'\n'

    return pattern_dict


def Euler_new(Graph):
    e = dict_to_edges(graph_to_dict(Graph))
    # e = Graph
    nodes = [v[0] for v in e]
    adj_nodes = [w[1] for w in e]
    edges = e[:]
    stack = []
    circuit = []
    n = nodes[0]
    stack.append(n)
    while len(circuit) != len(e):
        if n in [u[0] for u in edges]:
            n_index = nodes.index(n)
            adj_node = adj_nodes[n_index]
            tup = (n, adj_node)
            stack.append(adj_node)

            # edges.remove(tup)
            del edges[edges.index(tup)]
            # nodes.remove(n)
            del nodes[n_index]
            # adj_nodes.remove(adj_node)
            del adj_nodes[n_index]
            n = adj_node
        else:
            if len(stack) == 0:
                pass
            else:
                circuit.append(stack[-1])
                stack.pop(-1)
                n = stack[-1]

    r = circuit[::-1]
    # r.append(r[0])
    result = ''
    for i in r:
        result = result + i + '->'
    result = result[:-2]
    # return result
    return r


def AssemblingGenome(Pairs, k, d):
    db = DeBrujinFromPairs(Pairs)
    c = CycleToPath(db)
    r = SSBGP(c, k, d)
    return r


def indegree(v, e):
    i = [n[1] for n in e]
    return i.count(v)


def outdegree(v, e):
    o = [n[0] for n in e]
    return o.count(v)


def FindCycles(e):
    edges = e[:]
    nodes = [v[0] for v in e]
    adj_nodes = [w[1] for w in e]
    cycles = []
    n = nodes[0]
    cycle = []
    while len(nodes) > 0:

        n_index = nodes.index(n)
        cycle.append(n)
        adj_n = adj_nodes[n_index]
        del nodes[nodes.index(n)]
        del adj_nodes[n_index]

        if adj_n == cycle[0]:
            cycle.append(adj_n)
            cycles.append(cycle)
            cycle = []
            if len(nodes) > 0:
                n = nodes[0]
        else:
            n = adj_n
    return cycles


def MaximalNonBranchingPaths(Graph):
    Paths = []
    e = dict_to_edges(graph_to_dict(Graph))
    Nodes = [v[0] for v in e]
    nodes = [v[0] for v in e]
    adj_nodes = [w[1] for w in e]
    edges = e
    for node in Nodes:

        if (indegree(node, e) != 1) | (outdegree(node, e) != 1):

            if outdegree(node, e) > 0:

                node_index = nodes.index(node)
                NonBranchingPath = [(node, adj_nodes[node_index])]

                w = adj_nodes[node_index]
                del nodes[node_index]
                del adj_nodes[node_index]
                while indegree(w, e) == outdegree(w, e) == 1:
                    w_index = nodes.index(w)
                    NonBranchingPath.append((w, adj_nodes[w_index]))
                    w = adj_nodes[w_index]
                    del nodes[w_index]
                    del adj_nodes[w_index]
                else:
                    Paths.append(NonBranchingPath)

    if len(nodes) > 0:
        new_e = []
        for node in nodes:
            n_index = nodes.index(node)
            new_e.append((node, adj_nodes[n_index]))
        ic = FindCycles(new_e)
        Paths.extend(ic)

    q = ''
    for l in Paths:
        if type(l[0]) == tuple:

            s = ' -> '.join(l[0])
            for i in range(1, len(l)):
                s = s + ' -> ' + l[i][1]

        else:
            s = ' -> '.join(l)
        q = q + '\n' + s
    q = q[1:]
    return q


def StringReconstruction(Strings):
    result = []
    for i in Strings.split('\n'):
        path = i.split(' -> ')
        string = ''

        for j in range(1, len(path)):
            last_letter = path[j][-1]
            string = string + last_letter
            String = path[0] + string
        result.append(String)
    return result


def Contigs(Pattern):
    Db = DeBrujinFromKmers(Pattern)
    M = MaximalNonBranchingPaths(Db)
    S = StringReconstruction(M)

    print(*S)
