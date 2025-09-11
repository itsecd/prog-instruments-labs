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
    Files = []
else:
    Files = [samplefile]

Index = {} 
Taxon = {} 
coef = {} 
Taxon = {} 
taxon = []

pathLengths = {}

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
        Files.append(j)
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
            if x[Index[t]] == '/':
                # sample[species][t] = sample[species][t]
                sample[species][t] = mtax
            else:
                sample[species][t] = x[Index[t]]
                mtax = x[Index[t]]

            population[species][t] = sample[species][t]

    else:
        for t in taxon:
            # y = Taxon[t]
            sample[species][t] = x[Index[t]]
            population[species][t] = sample[species][t]

    # for t in taxon:
    # y = Taxon[t]
    #    population[species][t] = x[Index[t]]

if len(duplicates) > 0:
    print("Population master list contains duplicates:")
    for i in duplicates: print(i, '\n')


def path_length(population):
    taxonN = {}

    X = {}
    for t in taxon:
        Taxon[t] = {}
        X[t] = [population[i][t] for i in sample]

        if taxon.index(t) == 0:
            for i in set(X[t]):
                Taxon[t][i] = X[t].count(i)
        else:
            for i in set(X[t]):
                if i not in X[taxon[taxon.index(t) - 1]]:
                    Taxon[t][i] = X[t].count(i)

        taxonN[t] = len(Taxon[t])

    n = [float(len(Taxon[t])) for t in taxon]

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
    pathLengths = {}
    for i in range(len(taxon)):
        t = taxon[i]
        coef[t] = sum(adjco[i:])
        pathLengths[t] = adjco[i]

    return coef, taxonN, pathLengths


if pathlengths == 'n':
    coef, popN, pathLengths = path_length(population)
if pathlengths == 'y':
    XXX, popN, YYY = path_length(population)
    del XXX, YYY


# N = len(sample.keys())
def atd_mean(data, sample):
    # [sample = data.keys()
    N = len(sample)

    Taxon = {} 
    taxonN = {} 
    AvTD = 0 
    n = 0
    # Taxon are counts of taxa at each level, taxonN are numbers of
    # pairwise differences at each level, with n being the accumlation of
    # pairwise differences at that level. the difference between n and TaxonN
    # is the number of species that are in different taxa in that level
    # but not in upper levels

    for t in taxon:
        Taxon[t] = {}
        x = [data[i][t] for i in sample]
        for i in set(x):
            Taxon[t][i] = x.count(i)

    for t in taxon:
        taxonN[t] = sum([Taxon[t][i] * Taxon[t][j] for i in Taxon[t]
                                                  for j in Taxon[t] if i != j])
        n = taxonN[t] - n
        AvTD = AvTD + (n * coef[t])
        n = taxonN[t]

    # print sample
    AvTD /= (N * (N - 1))

    return AvTD, taxonN, Taxon


def atd_variance(taxonN, sample, atd):
    vtd = []

    # N = sum(taxon)

    vtd = 0 
    N = 0 
    n = 0

    for t in taxon:
        n = taxonN[t] - n
        vtd = vtd + n * coef[t] ** 2
        n = taxonN[t]

    N = len(sample)
    n = N * (N - 1)

    vtd = (vtd - ((atd * n) ** 2) / n) / n

    # vtd = (sum([tax1,tax2,tax3,tax4]) - (((atd * n)**2)/n))/n

    return vtd


def euler(data, atd, TaxonN):
    sample = data.keys()

    n = len(sample)
    TDmin = 0
    N = 0
    for t in taxon:
        k = len(Taxon[t])
        TDmin += coef[t] * (((k - 1) * (n - k + 1) * 2 +
                                 (k - 1) * (k - 2)) - N)
        N += ((k - 1) * (n - k + 1) * 2 + (k - 1) * (k - 2)) - N

    TDmin /= (n * (n - 1))

    # Taxon = {}

    # tax = []

    # taxon.append('sample')
    # Taxon['sample'] = sample
    taxon.reverse()
    TaxMax = {}
    taxonN = {}
    for t in taxon:
        TaxMax[t] = []
        if taxon.index(t) == 0:
            TaxMax[t] = []
            for i in range(len(Taxon[t])):
                TaxMax[t].append([])
            for i in range(len(Taxon[t])):
                TaxMax[t][i] = [sample[j] for j in range(i, n, len(Taxon[t]))]
        else:
            TaxMax[t] = []
            for i in range(len(Taxon[t])):
                TaxMax[t].append([])
                s = taxon[taxon.index(t) - 1]

                Tax = [TaxMax[s][j] for j in range(i, len(Taxon[s]),
                                                      len(Taxon[t]))]

                for j in Tax:
                    TaxMax[t][i] += j
        TaxMax[t].reverse()

    taxon.reverse() 
    TDmax = 0 
    n = 0 
    N = len(sample)
    for t in taxon:
        taxonN[t] = sum(
            [len(TaxMax[t][i]) * len(TaxMax[t][j])
            for i in range(len(TaxMax[t]))
            for j in range(len(TaxMax[t])) if
             i != j])
        n = taxonN[t] - n
        TDmax += n * coef[t]
        n = taxonN[t]
        # for i in TaxMax[t]:
        #    print t, len(i)

    TDmax /= (N * (N - 1))

    EI = (TDmax - atd) / (TDmax - TDmin)

    Eresults = {'EI': EI, 'TDmin': TDmin, 'TDmax': TDmax}
    return Eresults
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

for f in Files:
    sample = sample(f)
    f = f.split('.')
    f = f[0]

    results[f] = {}

    samp = sample.keys()

    atd, taxonN, Taxon = atd_mean(sample, samp)
    vtd = atd_variance(taxonN, samp, atd)
    Eresults = euler(sample, atd, taxonN)

    results[f]['atd'] = atd
    results[f]['vtd'] = vtd
    results[f]['euler'] = Eresults
    results[f]['N'] = taxonN
    results[f]['n'] = len(sample)
    results[f]['taxon'] = Taxon

N = len(sample.keys())
