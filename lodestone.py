#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------

'''
lodestone (let's hash books)

to generate book digests:
$ python lodestone.py --i path_to_books_dir --o digests

to analyze book digests:
$ python lodestone.py --score digests --num_variants x --graph
'''

#------------------------------------------------------------------------------

import argparse
import logging
import time
import os
import pickle
from pprint import pprint
from multiprocessing import Pool
import csv
import simhash
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

#------------------------------------------------------------------------------

LOG = logging.getLogger('lodestone')

#------------------------------------------------------------------------------

NAME_SEPARATOR = '__'

#------------------------------------------------------------------------------
    
def timeit(method):
    '''
    Simple profiling decorator
    '''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('Exec time for method --{}--: {:.2f} sec'.format(
                method.__name__,te-ts))
        return result

    return timed

#------------------------------------------------------------------------------
        
def hash_filepath_worker(param):
    '''
    Get fingerprint for given filepath. Use in dedicated worker process.
    '''
    filedesc, conf = param
    LOG.info('processing {}'.format(filedesc[0]))
    try:
        with open(filedesc[1], 'r') as fin:
            sh = simhash.simhash(fin.read(),
                                 k=conf['k'],
                                 lenhash=conf['lenhash'],
                                 remove_stopwords=conf['remove_stopwords'])
            return {'name': filedesc[0],
                    'sh': sh}
    except Exception:
        LOG.error('simhash', exc_info=True)
    return None
            
#------------------------------------------------------------------------------

def hash_path_async(path, conf, num_processes=20):
    '''
    Parallel calculation of fingerprints for all book files in the given
    directory.

    :param path: directory containing ebook files
    :param num_processes: number of processes to spawn
    '''
    texts = list(get_texts(path))
    pool = Pool(processes=num_processes)
    result = pool.map(hash_filepath_worker,
                      zip(texts, [conf]*len(texts)))
    pool.close()
    pool.join()
    return result

#------------------------------------------------------------------------------

def get_texts(indirs):
    '''
    Yields filepaths for .txt files in a given directory list indirs
    '''
    for indir in indirs:
        LOG.debug('Processing: {}'.format(indir))
        for filename in os.listdir(indir):
            filepath = os.path.join(indir, filename)
            # skip directories
            if os.path.isdir(filepath):
                continue
            name, ext = os.path.splitext(filename)
            if ext != '.txt':
                continue
            yield name, filepath

#------------------------------------------------------------------------------
            
def basename(fullname):
        '''
        let's assume fullname is always well formed
        '''
        return fullname.split(NAME_SEPARATOR)[0]

#------------------------------------------------------------------------------

def score_digests(digests, krange, num_variants, render_graph):
    '''
    Calculate distance for each pair of digests and score precision and recall
    '''
    if not num_variants:
        print('Must give --num_variants')
        return
    
    n = len(digests)
    # init distance matrix
    dmatrix = np.zeros(shape=(n,n))
    # calculate distance matrix, maybe we'll need it later
    for i in range(n):
        for j in range(n):
            dmatrix[i][j] = simhash.hamming(
                digests[i]['sh'], digests[j]['sh'])
    if not krange:
        krange = range(10, 50, 5)
    scores = []
    for k in krange:
        # number of true/false positives
        tp = 0
        fp = 0
        for i in range(n):
            similars = [j for j, d in enumerate(dmatrix[i])
                        if digests[j]['name'] != digests[i]['name'] and d <= k]
            for s in similars:
                if basename(digests[i]['name']) == basename(digests[s]['name']):
                    tp += 1
                else:
                    fp += 1
        if not tp and not fp:
            continue
        precision = float(tp)/(tp + fp)
        recall = float(tp)/(num_variants*n) # total number of duplicates
        scores.append((k, precision, recall))
    if render_graph:
        # precision graph
        pyt = [precision for _, precision, _ in scores]
        # recall graph
        ryt = [recall for _, _, recall in scores]
        xt = [k for k, _, _  in scores]
        plt.plot(pyt, 'r-')
        plt.plot(ryt, 'b-')
        plt.xticks(range(1, len(pyt)+1), xt)
        plt.show()
    return scores

#------------------------------------------------------------------------------

def cluster_digests(digests, conf, gold):
    '''
    Cluster digests using K-means algorithm

    :param digests: list of dicts with name and digest/sh attrs
    :param conf: hashing configuration
    :param gold: gold clusters dict
    '''
    # prepare data for clustering: extract every bit as a feature
    n = len(digests)
    lenhash = conf['lenhash']
    features = np.zeros(shape=(n, lenhash))
    for i in range(n):
        bin_str = bin(digests[i]['sh'])[2:]
        for j in range(lenhash):
            features[i][j] = float(bin_str[j])
    # number of clusters is equal to number of canonical texts
    n_clusters = len(set([basename(i['name']) for i in digests]))
    k_means = cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(features)
    labels = zip([i['name'] for i in digests], k_means.labels_)
    print('Num clusters: {}'.format(max(k_means.labels_) + 1))
    # calculate precision, recall and f1
    if gold_clusters:
        tp = fn = fp = 0.
        for i in range(n):
            for j in range(n):
                if labels[i][0] != labels[j][0]:
                    this_val = labels[i][1] == labels[j][1]
                    gold_val = gold[(labels[i][0], labels[j][0])]
                    # tp
                    if gold_val and this_val:
                        tp += 1
                    elif gold_val and not this_val:
                        fn += 1
                    elif not gold_val and this_val:
                        fp += 1
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        print('p={}, r={}, f1={}'.format(p, r, f1))

#------------------------------------------------------------------------------
    
if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s : %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',
                        help='path to input directory with txt books',
                        required=False,
                        nargs='*')
    parser.add_argument('--o',
                        help='path to output (pickled) file',
                        default='out.tmp',
                        required=False,
                        type=str)
    parser.add_argument('--k',
                        dest='conf_k',
                        help='k-gram',
                        default=1,
                        required=False,
                        type=int)
    parser.add_argument('--l',
                        dest='conf_lenhash',
                        help='length of the digest',
                        default=128,
                        required=False,
                        type=int)
    parser.add_argument('--remove_stopwords',
                        dest='conf_remove_stopwords',
                        help='if stopwords should be removed',
                        default=False,
                        required=False,
                        type=bool)
    parser.add_argument('--score',
                        help='input file for the score analysis',
                        default=False,
                        required=False,
                        type=str)
    parser.add_argument('--clusters',
                        dest='clusters',
                        help='input file for the cluster analysis',
                        default=None,
                        required=False,
                        type=str)
    parser.add_argument('--gold',
                        help='information about gold clusers',
                        default=None,
                        required=False,
                        type=str)
    parser.add_argument('--dist_k',
                        help='hamming distance to check',
                        default=None,
                        required=False,
                        type=int)
    parser.add_argument('--num_variants',
                        help='num of book variants',
                        default=None,
                        required=False,
                        type=int)
    parser.add_argument('--graph',
                        help='render graph',
                        default=False,
                        const=True,
                        nargs='?')
    args = parser.parse_args()
    conf = {'k': args.conf_k,
            'lenhash': args.conf_lenhash,
            'remove_stopwords': args.conf_remove_stopwords}
    if args.i:
        digests = hash_path_async(args.i, conf)
        digests = sorted(digests, key=lambda i: i['name'])
        with open(args.o, 'wb') as fout:
            pickle.dump({'digests': digests, 'conf': conf},
                        fout)
    elif args.score:
        with open(args.score, 'rb') as fin:
            krange = []
            if args.dist_k:
                krange = [args.dist_k]
            pprint(score_digests(pickle.load(fin),
                                 krange,
                                 args.num_variants,
                                 args.graph))
    elif args.clusters:
        gold_clusters = None
        if args.gold:
            with open(args.gold, 'rb') as fin:
                gold_clusters = pickle.load(fin)
        with open(args.clusters, 'rb') as fin:
            data = pickle.load(fin)
            cluster_digests(data['digests'], data['conf'], gold_clusters)
