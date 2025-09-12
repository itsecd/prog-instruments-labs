#!/usr/bin/python

"""
CODENAME:     PhyRe
DESCRIPTION:

Copyright (c) 2009 Ronald R. Ferrucci, Federico Plazzi, and Marco Passamonti..



Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

"""

import sys
from optparse import OptionParser
from random import sample
from re import match

P = 1000
D1 = 10
D2 = 70


samplefile = sys.argv[1] 
del sys.argv[1]
popfile = sys.argv[1] 
del sys.argv[1]
# outfile= sys.argv[1]       del sys.argv[1]

# outfile = samplefile
# output = open(outfile, 'w')

# efile = open('error.log','w')
# sys.stderr = efile

# output = open('output','w')
# out.write(allelesfile)
# out.close()

parser = OptionParser()

d1 = int(sys.argv[1]) 
del sys.argv[1]
d2 = int(sys.argv[1]) 
del sys.argv[1]

parser.add_option('-o')
parser.add_option('-p', type='int')
parser.add_option('-c')
parser.add_option('-b')
parser.add_option('-l')
parser.add_option('-m')

(options, args) = parser.parse_args()

if options.m:
    missing = options.m
else:
    missing = 'n'

if options.o:
    out = options.o
else:
    out = samplefile.split('.')[0]

if options.p:
    p = options.p
else:
    p = 1000

if options.c:
    ci = options.c
else:
    ci = 'y'

if options.b:
    batch = options.b
else:
    batch = 'n'

if options.l:
    pathlengths = options.l
else:
    pathlengths = 'n'

sample = {} 
population = {}

output = out + '.out'

o = open(output, 'a')

saveout = sys.stdout
sys.stdout = open(output, 'w')


# def taxon():
if batch == 'y':
    files = []
else:
    files = [samplefile]

index = {}
iaxon = {}
coef = {}
taxon = {}
taxon = []

path_lengths = {}

for i in open(samplefile):
    """
    if match('Taxon:', i):
        x = i.split()
        x.remove('Taxon:')
        #x = [string.lower() for string in x]  

        for i in x:
            taxon.append(i)
            j = x.index(i)
            Index[i] = j + 1
        continue

    elif match('Coefficients:', i):
        x = i.split()
        x.remove('Coefficients:')
        x = map(eval, x)

        for t in taxon:
            i = taxon.index(t)
            coef[t] = sum(x[i:])
            pathLengths[t] = x[i]

        continue
    """

    if batch == 'y':
        j = i.strip()
        files.append(j)
    else:
        break

duplicates = []

for i in open(popfile):
    if match('Taxon:', i):
        x = i.split()
        x.remove('Taxon:')
        # x = [string.lower() for string in x]

        for i in x:
            taxon.append(i)
            j = x.index(i)
            index[i] = j + 1
        continue

    elif match('Coefficients:', i):
        x = i.split()
        x.remove('Coefficients:')
        x = map(eval, x)

        for t in taxon:
            i = taxon.index(t)
            coef[t] = sum(x[i:])
            path_lengths[t] = x[i]

        continue

    i.strip()
    x = i.split()

    # if match('Taxon:', i): continue
    # if match('Coefficients:', i): continue

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
                # sample[species][t] = sample[species][t]
                sample[species][t] = mtax
            else:
                sample[species][t] = x[index[t]]
                mtax = x[index[t]]

            population[species][t] = sample[species][t]

    else:
        for t in taxon:
            # y = Taxon[t]
            sample[species][t] = x[index[t]]
            population[species][t] = sample[species][t]

    # for t in taxon:
    # y = Taxon[t]
    #    population[species][t] = x[Index[t]]

if len(duplicates) > 0:
    print("Population master list contains duplicates:")
    for i in duplicates: print(i, '\n')


def path_length(population):
    taxon_n = {}

    X = {}
    for t in taxon:
        taxon[t] = {}
        X[t] = [population[i][t] for i in sample]

        if taxon.index(t) == 0:
            for i in set(X[t]):
                taxon[t][i] = X[t].count(i)
        else:
            for i in set(X[t]):
                if i not in X[taxon[taxon.index(t) - 1]]:
                    taxon[t][i] = X[t].count(i)

        taxon_n[t] = len(taxon[t])

    n = [float(len(taxon[t])) for t in taxon]

    n.insert(0, 1.0)

    # s = 100/float(N)
    raw = []
    for i in range((len(n) - 1)):
        j = i + 1

        if n[i] > n[j]:
            c = 1
        else:
            c = (1 - n[i] / n[j])

        raw.append(c)

    s = sum(raw)
    adjco = [i * 100 / s for i in raw]

    coef = {}
    path_lengths = {}
    for i in range(len(taxon)):
        t = taxon[i]
        coef[t] = sum(adjco[i:])
        path_lengths[t] = adjco[i]

    return coef, taxon_n, pathLengths


if path_lengths == 'n':
    coef, popN, pathLengths = path_length(population)
if path_lengths == 'y':
    XXX, popN, YYY = path_length(population)
    del XXX, YYY


# N = len(sample.keys())
def atd_mean(data, sample):
    # [sample = data.keys()
    N = len(sample)

    taxon = {}
    taxon_n = {}
    avtd = 0
    n = 0
    # Taxon are counts of taxa at each level, taxonN are numbers of
    # pairwise differences at each level, with n being the accumlation of
    # pairwise differences at that level. the difference between n and TaxonN
    # is the number of species that are in different taxa in that level
    # but not in upper levels

    for t in taxon:
        taxon[t] = {}
        x = [data[i][t] for i in sample]
        for i in set(x):
            taxon[t][i] = x.count(i)

    for t in taxon:
        taxon_n[t] = sum([taxon[t][i] * taxon[t][j] for i in taxon[t]
                                                  for j in taxon[t] if i != j])
        n = taxon_n[t] - n
        avtd = avtd + (n * coef[t])
        n = taxon_n[t]

    # print sample
    avtd /= (N * (N - 1))

    return avtd, taxon_n, taxon


def atd_variance(taxon_n, sample, atd):
    vtd = []

    # N = sum(taxon)

    vtd = 0
    N = 0
    n = 0

    for t in taxon:
        n = taxon_n[t] - n
        vtd = vtd + n * coef[t] ** 2
        n = taxon_n[t]

    N = len(sample)
    n = N * (N - 1)

    vtd = (vtd - ((atd * n) ** 2) / n) / n

    # vtd = (sum([tax1,tax2,tax3,tax4]) - (((atd * n)**2)/n))/n

    return vtd


def euler(data, atd, taxon_n):
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

    # Taxon = {}

    # tax = []

    # taxon.append('sample')
    # Taxon['sample'] = sample
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
        taxon_n[t] = sum(
            [len(tax_max[t][i]) * len(tax_max[t][j])
            for i in range(len(tax_max[t]))
            for j in range(len(tax_max[t])) if
             i != j])
        n = taxon_n[t] - n
        td_max += n * coef[t]
        n = taxon_n[t]
        # for i in TaxMax[t]:
        #    print t, len(i)

    td_max /= (N * (N - 1))

    e_i = (td_max - atd) / (td_max - td_min)

    e_results = {'EI': e_i, 'TDmin': td_min, 'TDmax': td_max}
    return e_results
    # print TDmax


print("Output from Average Taxonomic Distinctness\n")


def sample(samplefile):
    sample = {}
    print(samplefile)
    for i in open(samplefile):
        if match('Taxon:', i):
            continue
        elif match('Coefficients:', i):
            continue

        x = i.split()

        species = x[0]
        # sample[species] = {}

        sample[species] = population[species]

    return sample


results = {}

for f in files:
    sample = sample(f)
    f = f.split('.')
    f = f[0]

    results[f] = {}

    samp = sample.keys()

    atd, taxon_n, taxon = atd_mean(sample, samp)
    vtd = atd_variance(taxon_n, samp, atd)
    e_results = euler(sample, atd, taxon_n)

    results[f]['atd'] = atd
    results[f]['vtd'] = vtd
    results[f]['euler'] = e_results
    results[f]['N'] = taxon_n
    results[f]['n'] = len(sample)
    results[f]['taxon'] = taxon_n

N = len(sample.keys())
