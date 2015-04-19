#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------

'''
lodestone (let's hash books)

to generate book digests:
$ python lodestone.py --i path_to_books_dir --o digests

to analyze book digests:
$ python lodestone.py --analyze digests
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
from sklearn.cluster import DBSCAN, AffinityPropagation
from sklearn import metrics

#------------------------------------------------------------------------------

LOG = logging.getLogger('lodestone')

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

def score_digests(digests):
    '''
    Calculate distance for each pair of digests and score precision and recall
    '''
    n = len(digests)
    # init distance matrix
    dmatrix = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            dmatrix[i][j] = simhash.hamming(digests[i]['sh'], digests[j]['sh'])
            #print(digests[i]['name'], digests[j]['name'], dmatrix[i][j])
    for i in range(n):
        dmatrix[i] = sorted(dmatrix[i])
    print dmatrix

#------------------------------------------------------------------------------
    
if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s : %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',
                        dest='i',
                        help='path to input directory with txt books',
                        required=False,
                        nargs='*')
    parser.add_argument('--o',
                        dest='o',
                        help='path to output (pickled) file',
                        default='out.pickle',
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
    parser.add_argument('--analyze',
                        dest='analyze',
                        help='input file for the score analysis',
                        default=False,
                        required=False,
                        type=str)
    args = parser.parse_args()
    conf = {'k': args.conf_k,
            'lenhash': args.conf_lenhash,
            'remove_stopwords': args.conf_remove_stopwords}
    if args.i:
        digests = hash_path_async(args.i, conf)
        digests = sorted(digests, key=lambda i: i['name'])
        with open(args.o, 'wb') as fout:
            pickle.dump(digests, fout)
    elif args.analyze:
        with open(args.analyze, 'rb') as fin:
            score_digests(pickle.load(fin))
