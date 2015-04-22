#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------

'''
lodestone (let's hash books)

to generate book digests:
$ python lodestone.py --i path_to_books_dir --o digests

--k
--l
--remove_stopwords

to analyze book digests:
$ python lodestone.py --clusters digests --gold gold_clusters_file

to do a baseline analysis:
$ python lodestone.py --baseline path_to_books_dir --gold gold_clusters_file
'''

#------------------------------------------------------------------------------

import argparse
import logging
import time
import os
import pickle
from pprint import pprint
from multiprocessing import Pool, Process, Queue
import csv
import simhash
import numpy as np
#import matplotlib.pyplot as plt
from sklearn import cluster
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import utils

#------------------------------------------------------------------------------

LOG = logging.getLogger('lodestone')

#------------------------------------------------------------------------------

NAME_SEPARATOR = '__'

#------------------------------------------------------------------------------

def hash_filepath_worker(filedesc, conf):
    '''
    Get fingerprint for given filepath. Use in dedicated worker process.
    '''
    LOG.info('processing {}'.format(filedesc))
    with open(filedesc[1], 'r') as fin:
        sh = simhash.simhash(fin.read(),
                             k=conf['k'],
                             lenhash=conf['lenhash'],
                             remove_stopwords=conf['remove_stopwords'])
        return {'name': filedesc[0], 'sh': sh}
    return None
            
#------------------------------------------------------------------------------

@utils.timeit
def hash_path_async(path, conf):
    '''
    Parallel calculation of fingerprints for all book files in the given
    directory.

    :param path: directory containing ebook files
    :param num_processes: number of processes to spawn
    '''
    # sequential
    return [hash_filepath_worker(text_name, conf)
            for text_name in get_texts(path)]
    # parallel pool
    #pool = Pool(processes=8)
    #results = [pool.apply_async(hash_filepath_worker, args=(text, conf))
    #           for text in get_texts(path)]
    #output = [p.get() for p in results]
    #return output
    # parallel workers
    # workers = 10
    # procs = []
    # inqueue = Queue()
    # outqueue = Queue()
    # for text_name in get_texts(path):
    #     inqueue.put(text_name)
    # for w in xrange(workers):
    #     p = Process(target=hash_filepath_worker, args=(w, conf, inqueue, outqueue))
    #     p.start()
    #     procs.append(p)
    #     inqueue.put('/')
    # [p.join() for p in procs]
    # outqueue.put('/')
    # output = [result for result in iter(outqueue.get, '/')]
    # return output

#------------------------------------------------------------------------------

def get_texts(indirs):
    '''
    Yields filepaths for .txt files in a given directory list indirs

    :param indirs: directory (wildcard) where to look for books
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

def calc_scores(labels, indir):
    '''
    Calculate precision, recall and f1 by pairwise comparison of clustered
    labels and gold clusters pairs
    '''
    with open(os.path.join(indir[0], 'gold_clusters.tmp')) as fin:
        gold = pickle.load(fin)
        n = len(labels)
        tp = fn = fp = 0.
        for i in range(n):
            for j in range(n):
                if labels[i][0] != labels[j][0]:
                    this_val = labels[i][1] == labels[j][1]
                    gold_val = gold[(labels[i][0], labels[j][0])]
                    if gold_val and this_val:
                        tp += 1
                    elif gold_val and not this_val:
                        fn += 1
                    elif not gold_val and this_val:
                        fp += 1
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
        print('p={:.2f}, r={:.2f}, f1={:.2f}'.format(p, r, f1))

#------------------------------------------------------------------------------

def cluster_digests(digests, conf, indir):
    '''
    Cluster digests using K-means algorithm

    :param digests: list of dicts with name and digest/sh attrs
    :param conf: hashing configuration
    :param indir: directory (wildcard) where to look for books
    '''
    # prepare data for clustering: extract every bit as a feature
    n = len(digests)
    lenhash = conf['lenhash']
    features = np.zeros(shape=(n, lenhash))
    for i in range(n):
        bin_str = format(digests[i]['sh'], '0{}b'.format(conf['lenhash']))
        for j in range(lenhash):
            features[i][j] = float(bin_str[j])
    # number of clusters is equal to number of canonical texts
    n_clusters = len(set([basename(i['name']) for i in digests]))
    k_means = cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(features)
    labels = zip([i['name'] for i in digests], k_means.labels_)
    print('Num clusters: {}'.format(max(k_means.labels_) + 1))
    calc_scores(labels, indir)

#------------------------------------------------------------------------------
        
def run_baseline(indir):
    '''
    Calculates scores based on baseline bag of words algorithm

    :param indir: directory (wildcard) where to look for books
    '''
    stopwords = nltk.corpus.stopwords.words('english')
    files = sorted(list(get_texts(indir)), key=lambda x: x[0])
    vectorizer = TfidfVectorizer(input='filename',
                                 decode_error='ignore',
                                 max_features=100000,
                                 max_df=0.8,
                                 min_df=0.2,
                                 #stop_words='english',
                                 #use_idf=True,
                                 )
    features = vectorizer.fit_transform([f[1] for f in files])
    #tfidf_vectorizer.get_feature_names()
    # number of clusters is equal to number of canonical texts
    n_clusters = len(set([basename(f[0]) for f in files]))
    k_means = cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(features)
    labels = zip([f[0] for f in files], k_means.labels_)
    print('Num clusters: {}'.format(max(k_means.labels_) + 1))
    calc_scores(labels, indir)

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
                        help='input file for the cluster analysis',
                        default=None,
                        required=False,
                        type=str)
    parser.add_argument('--gold',
                        help='information about gold clusers',
                        default=None,
                        required=False,
                        type=str)
    parser.add_argument('--baseline',
                        help='input directory for the baseline analysis',
                        default=None,
                        required=False,
                        nargs='*')
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
        with open(args.clusters, 'rb') as fin:
            data = pickle.load(fin)
            cluster_digests(data['digests'], data['conf'], args.clusters)
    elif args.baseline:
        run_baseline(args.baseline)
