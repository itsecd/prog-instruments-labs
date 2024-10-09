import sys
from optparse import OptionParser
from re import match
from random import sample

def initialize_parameters():
    """Initializing parameters from the command line."""
    parser = OptionParser()
    parser.add_option('-o')
    parser.add_option('-p', type='int')
    parser.add_option('-c')
    parser.add_option('-b')
    parser.add_option('-l')
    parser.add_option('-m')

    d1 = int(sys.argv[1])
    del sys.argv[1]
    d2 = int(sys.argv[1])
    del sys.argv[1]

    (options, args) = parser.parse_args()
    out = options.o if options.o else sample_file.split('.')[0]
    p = options.p if options.p else 1000
    ci = options.c if options.c else 'y'
    batch = options.b if options.b else 'n'
    path_lengths = options.l if options.l else 'n'
    missing = options.m if options.m else 'n'

    return sample_file, pop_file, p, d1, d2, ci, batch, path_lengths, missing, out

def read_population_file(pop_file, sample, missing, batch, sample_file):
    """Reading the population file and returning the data."""
    population = {}
    taxon = []
    coef = {}
    path_lengths_dict = {}
    index = {}

    output = sample_file.split('.')[0] + '.out'
    with open(output, 'a') as o:
        save_out = sys.stdout
        sys.stdout = o  

        files = []
        if batch == 'y':
            with open(sample_file) as sf:
                for line in sf:
                    files.append(line.strip())
        else:
            files = [sample_file]

        duplicates = []

        with open(pop_file) as pf:
            for line in pf:
                line = line.strip()
                
                if match('Taxon:', line):
                    x = line.split()
                    x.remove('Taxon:')
                    for i in x:
                        taxon.append(i)
                        index[i] = x.index(i) + 1
                    continue

                elif match('Coefficients:', line):
                    x = line.split()
                    x.remove('Coefficients:')
                    x = list(map(eval, x))

                    for t in taxon:
                        i = taxon.index(t)
                        coef[t] = sum(x[i:])
                        path_lengths_dict[t] = x[i]
                    continue

                x = line.split()
                species = x[0]
                population[species] = {}

                if species in sample.keys():
                    duplicates.append(species)
                else:
                    sample[species] = {}
                    population[species] = {}

                if missing == 'y':
                    mtax = ''
                    for t in taxon:
                        if x[index[t]] == '/':
                            sample[species][t] = mtax
                        else:
                            sample[species][t] = x[index[t]]
                            mtax = x[index[t]]

                        population[species][t] = sample[species][t]
                else:
                    for t in taxon:
                        sample[species][t] = x[index[t]]
                        population[species][t] = sample[species][t]

        if duplicates:
            print("Population master list contains duplicates:")
            for i in duplicates:
                print(i)

    sys.stdout = save_out

    return population, taxon, coef, path_lengths_dict

def read_sample_file(sample_file):
    """Reading the selection file and returning the data."""
    sample = {}
    for line in open(sample_file):
        if match('Taxon:', line) or match('Coefficients:', line):
            continue
        x = line.split()
        species = x[0]
        sample[species] = {}
    return sample

def path_length(population):
    """Calculation of the path length."""
    taxon_n = {}
    x = {}
    for t in taxon:
        taxon[t] = {}
        x[t] = [population[i][t] for i in sample]

        if taxon.index(t) == 0:
            for i in set(x[t]):
                taxon[t][i] = x[t].count(i)
        else:
            for i in set(x[t]):
                if i not in x[taxon[taxon.index(t) - 1]]:
                    taxon[t][i] = x[t].count(i)

        taxon_n[t] = len(taxon[t])

    n = [float(len(taxon[t])) for t in taxon]
    n.insert(0, 1.0)
    raw = []
    for i in range(len(n) - 1):
        j = i + 1

        if n[i] > n[j]:
            c = 1
        else:
            c = (1 - n[i] / n[j])

        raw.append(c)

    s = sum(raw)
    adj_co = [i * 100 / s for i in raw]

    coef = {}
    path_lengths = {}
    for i in range(len(taxon)):
        t = taxon[i]
        coef[t] = sum(adj_co[i:])
        path_lengths[t] = adj_co[i]

    return coef, taxon_n, path_lengths

def atd_mean(data: dict, sample: list) -> tuple:
    """Calculates the average taxonomic distinctness."""
    N = len(sample)
    taxon = {}
    taxon_n = {}
    av_td = 0
    n = 0

    for t in taxon:
        taxon[t] = {}
        x = [data[i][t] for i in sample]
        for i in set(x):
            taxon[t][i] = x.count(i)

    for t in taxon:
        taxon_n[t] = sum([taxon[t][i] * taxon[t][j]
                          for i in taxon[t] for j in taxon[t] if i != j])
        n = taxon_n[t] - n
        av_td += n * coef[t]
        n = taxon_n[t]
    av_td /= (N * (N - 1))
    return av_td, taxon_n, taxon

def atd_variance(taxon_n: dict, sample: list, atd: float) -> float:
    """Calculates the variance of taxonomic distinctness."""
    v_td = 0
    N = 0
    n = 0

    for t in taxon:
        n = taxon_n[t] - n
        v_td += n * coef[t] ** 2
        n = taxon_n[t]

    N = len(sample)
    n = N * (N - 1)
    v_td = (v_td - ((atd * n) ** 2) / n) / n

    return v_td

def euler(data: dict, atd: float, taxon_n: dict) -> dict:
    """Calculates Euler index for the data."""
    sample = data.keys()
    n = len(sample)
    td_min = 0
    N = 0
    for t in taxon:
        k = len(taxon[t])
        td_min += coef[t] * (((k - 1) * (n - k + 1) * 2 +
                              (k - 1) * (k - 2)) - N)
        N += ((k - 1) * (n - k + 1) * 2 + (k - 1) * (k - 2)) - N
    td_min /= (n * (n - 1))

    taxon.reverse()
    tax_max = {}
    taxon_n = {}

    for t in taxon:
        tax_max[t] = []
        if taxon.index(t) == 0:
            tax_max[t] = []
            for i in range(len(taxon[t])):
                tax_max[t].append([])
            for i in range(len(taxon[t])):
                tax_max[t][i] = [sample[j] for j in range(i, n, len(taxon[t]))]
        else:
            tax_max[t] = []
            for i in range(len(taxon[t])):
                tax_max[t].append([])
                s = taxon[taxon.index(t) - 1]
                tax = [tax_max[s][j] for j in range(i, len(taxon[s]),
                                                    len(taxon[t]))]

                for j in tax:
                    tax_max[t][i] += j
        tax_max[t].reverse()

    taxon.reverse()
    td_max = 0
    n = 0
    N = len(sample)
    for t in taxon:
        taxon_n[t] = sum([len(tax_max[t][i]) * len(tax_max[t][j])
                          for i in range(len(tax_max[t]))
                          for j in range(len(tax_max[t])) if i != j])
        n = taxon_n[t] - n
        td_max += n * coef[t]
        n = taxon_n[t]

    td_max /= (N * (N - 1))

    ei = (td_max - atd) / (td_max - td_min)

    e_results = {'EI': ei, 'TDmin': td_min, 'TDmax': td_max}
    return e_results

def print_results(results, pop_n, path_lengths_dict) -> None:
    """Prints the analysis results to the screen."""
    print("Number of taxa and path lengths for each taxonomic level:")

    for t in taxon:
        print('%-10s\t%d\t%.4f' % (t, pop_n[t], path_lengths_dict[t]))

    print("\n")

    for f in results:
        print("---------------------------------------------------")
        print("Results for sample: ", f, '\n')
        print("Dimension for this sample is", results[f]['n'], '\n\n')
        print("Number of taxa and pairwise comparisons at each taxon level:")

        n = 0
        for t in taxon:
            N = results[f]['N'][t] - n
            print('%-10s\t%i\t%i' % (t, len(results[f]['taxon'][t]), N))
            n = results[f]['N'][t]

        print("\nAverage taxonomic distinctness      = %.4f" % results[f]['atd'])
        print("Variation in taxonomic distinctness = %.4f" % results[f]['vtd'])
        print("Minimum taxonomic distinctness      = %.4f" % results[f]['euler']['TDmin'])
        print("Maximum taxonomic distinctness      = %.4f" % results[f]['euler']['TDmax'])
        print("von Euler's index of imbalance      = %.4f" % results[f]['euler']['EI'])
        print('\n')