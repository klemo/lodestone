#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------

'''
lodestone (let's hash books)

to generate book digests:
$ python lodestone.py --i path_to_books_dir --o digests

additional params: --k --l --stopwords

to analyze book digests using kmeans clustering:
$ python lodestone.py --clusters digests --gold gold_clusters_file [-l]

to do a baseline analysis:
$ python lodestone.py --baseline path_to_books_dir --gold gold_clusters_file

to do a query score analysis:
$ python lodestone.py --score path_to_digests_csv --gold gold_clusters_file
'''

#------------------------------------------------------------------------------

import argparse
import logging
import time
import os
import sys
import pickle
import csv
import simhash
import simplejson
import numpy as np
import nltk
import pika
import uuid
import utils
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pprint import pprint

#------------------------------------------------------------------------------

LOG = logging.getLogger('lodestone')

#------------------------------------------------------------------------------

Q = 'lodestone_q'
NAME_SEPARATOR = '__'

#------------------------------------------------------------------------------
            
def basename(fullname):
        '''
        let's assume fullname is always well formed:
        basenameNAME_SEPARATORsuffix
        '''
        return fullname.split(NAME_SEPARATOR)[0]

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

@utils.timeit
def hash_path_async(path, conf):
    '''
    Parallel calculation of fingerprints for all book files in the given
    directory.

    :param path: directory containing ebook files
    :param num_processes: number of processes to spawn
    '''

    output = [] # will populate this asynchronously with digests
    corr_ids = [] # correlation ids for pika messages
    
    def on_response(ch, method, props, body):
        '''
        Pika callback: check correlation_id and append to output
        '''
        if props.correlation_id in corr_ids:
            response = simplejson.loads(body)
            print response
            output.append(response)

    def submit_task(text_name):
        '''
        Submit taks to remote worker
        '''
        corr_id = str(uuid.uuid4())
        corr_ids.append(corr_id)
        channel.basic_publish(
            exchange='',
            routing_key=Q,
            properties=pika.BasicProperties(reply_to=callback_queue,
                                            correlation_id=corr_id),
            body=simplejson.dumps((text_name, conf)))

    LOG.info('Conf: {}'.format(conf))
    # setup basic rabbit rpc
    connection = pika.BlockingConnection(pika.ConnectionParameters(
            host='localhost'))
    channel = connection.channel()
    result = channel.queue_declare(exclusive=True)
    callback_queue = result.method.queue
    channel.basic_consume(on_response, no_ack=True,
                          queue=callback_queue)
    # get list of text files in the dir path
    texts = list(get_texts(path))
    # submit to workers
    [submit_task(text) for text in texts]
    # collect digests
    while len(output) < len(texts):
        connection.process_data_events()
    return output
    # sequential version
    # for text_name in get_texts(path):
    #     LOG.info('processing {}'.format(filedesc))
    #     with open(filedesc[1], 'r') as fin:
    #         sh = simhash.simhash(fin.read(),
    #                              k=conf['k'],
    #                              lenhash=conf['lenhash'],
    #                              stopwords=conf['stopwords'])
    #         output.append({'name': filedesc[0], 'sh': sh})
    # return output

#------------------------------------------------------------------------------

def score_digests(digests, num_variants, max_bits, render_graph):
    '''
    Calculate distance for each pair of digests and score precision and recall

    :param digests: list of name,sh objects
    :param num_variants: number of duplicates for each book
    :param render_graph: if graph should be rendered
    '''
    if not num_variants:
        print('Must give --num_variants')
        return

    n = len(digests)
    print('Calculating distance matrix for {} digests'.format(n))
    # init distance matrix ----------------------------------------------------
    dmatrix = np.zeros(shape=(n,n))
    # calculate distance matrix, maybe we'll need it later
    for i in range(n):
        sys.stdout.write('\r{}/{}'.format(i+1, n))
        sys.stdout.flush()
        for j in range(n):
            dmatrix[i][j] = simhash.hamming(
                digests[i]['sh'], digests[j]['sh'])
    print('\n')
    # calc prec/recall for different k ----------------------------------------
    scores = []
    krange = range(0, max_bits, 1)
    for k in krange:
        # number of true/false positives
        all_p = []
        all_r = []
        for i in range(n):
            tp = fp = 0.
            similars = [j for j, d in enumerate(dmatrix[i])
                        if digests[j]['name'] != digests[i]['name'] and d <= k]
            for s in similars:
                if basename(digests[i]['name']) == basename(digests[s]['name']):
                    tp += 1
                else:
                    fp += 1
            if not tp and not fp:
                #LOG.info('No results for k={}'.format(k))
                continue
                #p = 1.
                #r = 0.
            else:
                p = tp / (tp + fp)
                r = tp / num_variants
            all_p.append(p)
            all_r.append(r)
        avg_p = np.mean(all_p)
        avg_r = np.mean(all_r)
        f1 = 2*avg_p*avg_r/(avg_p+avg_r)
        print('k={}: p={:.2f}, r={:.2f}, f1={:.2f}'.format(
                k, avg_p, avg_r, f1))
        scores.append((k, avg_p, avg_r, f1))
    #--------------------------------------------------------------------------
    if render_graph:
        import matplotlib.pyplot as plt
        print('Max f1: {:.2f}'.format(max([s[3] for s in scores])))
        xt = [s[0] for s  in scores] # x-axis max k bits
        pyt = [s[1] for s in scores] # precision graph
        ryt = [s[2] for s in scores] # recall graph
        line_precision, = plt.plot(pyt, 'r-', label='precision', linestyle='--')
        line_recall, = plt.plot(ryt, 'b-', label='recall')
        plt.legend()
        plt.xticks(np.arange(min(xt), max(xt)+1, 15.0))
        plt.xlabel('k-bit distance')
        plt.ylabel('Precision/recall')
        plt.title('Precision/recall for {} books'.format(n))
        plt.show()
    return scores

#------------------------------------------------------------------------------

def calc_scores(labels, gold):
    '''
    Calculate precision, recall and f1 by pairwise comparison of clustered
    labels and gold clusters pairs
    '''
    n = len(labels)
    tp = fn = fp = 0.
    seen = set()
    for i in range(n):
        for j in range(n):
            if labels[i][0] != labels[j][0]:
                key = sorted((labels[i][0], labels[j][0]))
                skey = ', '.join(key)
                if skey in seen:
                    continue
                else:
                    seen.add(skey)
                    gold_val = gold[skey]
                    this_val = labels[i][1] == labels[j][1]
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
        bin_str = format(digests[i]['sh'], '0{}b'.format(conf['lenhash']))
        for j in range(lenhash):
            features[i][j] = float(bin_str[j])
    # number of clusters is equal to number of canonical texts
    n_clusters = len(set([basename(i['name']) for i in digests]))
    # initial centers are canonical texts themselves
    initial_centers = np.zeros(shape=(n_clusters, lenhash))
    ci = 0
    for i in range(n):
        if digests[i]['name'] == basename(digests[i]['name']) + '__':
            initial_centers[ci] = features[i]
            ci += 1
    k_means = cluster.KMeans(n_clusters=n_clusters, n_jobs=3,
                             init=initial_centers)
    k_means.fit(features)
    labels = zip([i['name'] for i in digests], k_means.labels_)
    print('Num clusters: {}'.format(max(k_means.labels_) + 1))
    calc_scores(labels, gold)

#------------------------------------------------------------------------------

def run_baseline(indir, gold):
    '''
    Calculates scores based on baseline bag of words algorithm

    :param indirs: directory (wildcard) where to look for books
    :param gold: gold clusters dict
    '''
    stopwords = nltk.corpus.stopwords.words('english')
    files = sorted(list(get_texts(indir)), key=lambda x: x[0])
    vectorizer = TfidfVectorizer(input='filename',
                                 decode_error='ignore',
                                 max_features=20000,
                                 stop_words='english',
                                 use_idf=False,
                                 )
    features = vectorizer.fit_transform([f[1] for f in files])
    # number of clusters is equal to number of canonical texts
    n_clusters = len(set([basename(f[0]) for f in files]))
    k_means = cluster.KMeans(n_clusters=n_clusters, n_jobs=3)
    k_means.fit(features)
    labels = zip([f[0] for f in files], k_means.labels_)
    print('Num clusters: {}'.format(max(k_means.labels_) + 1))
    calc_scores(labels, gold)

#------------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s : %(message)s',
                        level=logging.INFO)
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
                        nargs=2,
                        default=['3', '4'])
    parser.add_argument('--l',
                        dest='conf_lenhash',
                        help='length of the digest',
                        default=128,
                        required=False,
                        type=int)
    parser.add_argument('--stopwords',
                        dest='conf_stopwords',
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
    parser.add_argument('--num_variants',
                        help='num of book variants',
                        default=None,
                        required=False,
                        type=int)
    parser.add_argument('--max_bits',
                        help='hamming bit range',
                        default=48,
                        required=False,
                        type=int)
    parser.add_argument('--graph',
                        help='render graph',
                        default=False,
                        const=True,
                        nargs='?')
    args = parser.parse_args()
    conf = {'k': [int(i) for i in args.conf_k],
            'lenhash': args.conf_lenhash,
            'stopwords': args.conf_stopwords}
    #--------------------------------------------------------------------------
    if args.i:
        digests = hash_path_async(args.i, conf)
        digests = sorted(digests, key=lambda i: i['name'])
        with open(args.o, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ')
            for digest in digests:
                # name hex_digest
                csvwriter.writerow([digest['name'],
                                    hex(digest['sh']).strip('L')])
    #--------------------------------------------------------------------------
    elif args.score:
        with open(args.score, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ')
            digests = []
            for row in csvreader:
                digests.append({'name': row[0], 'sh': int(row[1], 16)})
            score_digests(
                digests, args.num_variants, args.max_bits, args.graph)
    #--------------------------------------------------------------------------
    elif args.clusters and args.gold and conf:
        with open(args.clusters, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=' ')
            digests = []
            for row in csvreader:
                digests.append({'name': row[0], 'sh': int(row[1], 16)})
            with open(args.gold, 'rb') as fin:
                gold_clusters = pickle.load(fin)
                cluster_digests(digests, conf, gold_clusters)
    #--------------------------------------------------------------------------
    elif args.baseline:
        with open(
            os.path.join(args.baseline[0], 'gold_clusters.tmp'), 'r') as fin:
            run_baseline(args.baseline, pickle.load(fin))
