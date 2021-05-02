def is_sorted(k, p):
    if p[abs(k) - 1] == abs(k):
        return True
    else:
        return False


def k_sorting_reversal(k, p):
    n = p[:]
    k_index = [abs(v) for v in p].index(abs(k))
    if k_index + 1 > abs(k):
        sublist = (p[abs(k) - 1:k_index + 1][::-1])
        sublist = [v * (-1) for v in sublist]
        n[k - 1:k_index + 1] = sublist
    else:
        sublist = (p[k_index:abs(k)][::-1])
        sublist = [v * (-1) for v in sublist]
        n[k_index:k] = sublist
    return n


def greedy_sorting(p):
    p_new = p[:]
    steps = []
    for i in range(1, len(p_new) + 1):
        if is_sorted(i, p_new) == False:
            p_new = k_sorting_reversal(i, p_new)
            steps.append(p_new)
        if p_new[i - 1] == -i:
            p_new = k_sorting_reversal(i, p_new)
            steps.append(p_new)
    return steps


def num_of_breakpoints(p):
    np = p[:]
    c = 0
    np.append(len(np) + 1)
    np = [0] + np
    for i in range(len(np) - 1):
        if np[i + 1] - np[i] != 1:
            c += 1
    return c


def chromosome_to_cycle(chromosome):
    cycle = []

    for i in chromosome:
        if i > 0:
            cycle.append((2 * i) - 1)
            cycle.append(2 * i)
        else:
            cycle.append(abs(2 * i))
            cycle.append(abs((2 * i)) - 1)
    return cycle


def cycle_to_chromosome(cycle):
    chrom = []
    for i in range(int(len(cycle) / 2)):
        if cycle[(i * 2) + 1] > cycle[i * 2]:
            chrom.append(int(cycle[(i * 2) + 1] / 2))
        else:
            chrom.append(int(-1 * (cycle[i * 2]) / 2))
    return chrom


def format_chromosome(chromosome):
    p = chromosome.split(sep='(')
    p = [v[:-1] for v in p]
    p = [v.split() for v in p]
    p = [[int(v) for v in group] for group in p]
    b = []
    for i in p[1:]:
        for j in i:
            b.append(j)
    return p[1:]


def colored_edges(chromosome):
    p = format_chromosome(chromosome)
    edges = []
    for c in p:
        nodes = chromosome_to_cycle(c)
        for i in range(1, len(nodes) - 1, 2):
            edges.append((nodes[i], nodes[i + 1]))
        edges.append((nodes[-1], nodes[0]))
    return edges


def find_cycles(graph):
    b = []
    chromosome = []
    for i in range(1, len(graph) - 1, 2):
        if abs(graph[i] - graph[i + 1]) != 1:
            b.append(i + 1)
    cycles = [graph[i:j] for i, j in zip([0] + b, b + [None])]
    for i in range(len(cycles)):
        cycles[i] = [cycles[i][-1]] + cycles[i][:-1]
    for i in cycles:
        chromosome.append(tuple(cycle_to_chromosome(i)))
    return chromosome


def get_edges(chromosomes):
    c_final = []
    c = chromosomes.split(sep='\n')
    for i in c:
        c_final.append(colored_edges(i))
    edges = [item for sublist in c_final for item in sublist]
    return edges


def get_cycles(graph):
    g = {}
    for edge in graph:
        (v, w) = edge
        if v not in g:
            g[v] = [w]
        else:
            g[v].append(w)
        if w not in g:
            g[w] = [v]
        else:
            g[w].append(v)
    unvisited = set(g.keys())
    cycles = []

    while len(unvisited) > 1:

        start = min(unvisited)
        v = start
        unvisited.remove(start)
        cycle = []
        while True:
            next_w = None
            for w in g[v]:
                if w in unvisited:
                    next_w = w
                    break
            if next_w == None:
                cycle.append((v, start))
                break
            cycle.append((v, next_w))
            v = next_w

            unvisited.remove(next_w)
        cycles.append(cycle)
    return cycles


def two_break_distance(chromosome):
    edge = get_edges(chromosome)
    cycles = get_cycles(edge)
    blocks = chromosome.split(sep='\n')
    return int((len(edge) / 2) - len(cycles))


def two_break_on_genome_graph(graph, i1, i2, i3, i4):
    # graph_list = list(make_tuple(graph))
    graph_list = [set(v) for v in graph]
    graph_list.remove({i1, i2})
    graph_list.remove({i3, i4})
    graph_list.append({i1, i3})
    graph_list.append({i2, i4})
    graph_list = [tuple(v) for v in graph_list]
    return graph_list


def two_break_on_genome(P, i1, i2, i3, i4):
    genome_graph = get_edges(P)
    genome_graph = two_break_on_genome_graph(genome_graph, i1, i2, i3, i4)
    genome_graph_f = []
    for i in genome_graph:
        for j in i:
            genome_graph_f.append(j)
    chromosome = find_cycles(genome_graph_f)
    return genome_graph_f


def shared_kmers(k, s1, s2):
    kmer_dict = {}
    positions = []
    for i in range(len(s1) - k + 1):
        first_kmer = s1[i:i + k]
        if first_kmer in kmer_dict:
            kmer_dict[first_kmer].append(i)
        else:
            kmer_dict[first_kmer] = [i]
    for j in range(len(s2) - k + 1):
        second_kmer = s2[j:j + k]
        if second_kmer in kmer_dict:
            positions.append((kmer_dict[second_kmer], j))
        if reverse_compliment(second_kmer) in kmer_dict:
            positions.append((kmer_dict[reverse_compliment(second_kmer)], j))
    return positions
