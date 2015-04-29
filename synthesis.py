#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------

'''
lodestone (let's hash books)

generates synthetic text dataset by corrupting referent dataset with random
errors; errors simulate ocr confusions and random edits

to generate synthetic dataset:
$ python synthesis.py --gen --i indir --o outdir [--gold]

to write only gold clusters membership pairs:
$ python synthesis.py --gold --i indir
'''

#------------------------------------------------------------------------------

import argparse
import logging
import os
from pprint import pprint
import pickle
from multiprocessing import Pool
import numpy
from scipy import stats
import random
import string
import shutil
from collections import defaultdict
import utils
import nltk

#------------------------------------------------------------------------------

LOG = logging.getLogger(__name__)

#------------------------------------------------------------------------------

NAME_SEPARATOR = '__'
LETTERS = string.ascii_letters + string.digits

#------------------------------------------------------------------------------

# (insert|delete|edit, char, probability)
char_mutations = [
    ('i',   ord(' '),            0.32), # space insertion
    ('d',   ord(' '),            0.05), # space deletion
    ('i',   None,                0.04), # any char insertion
    ('d',   None,                0.05), # any char deletion
    ('e',   (1, 1),              0.25), # 1:1 edit
    ('e',   (1, 2),              0.06), # 1:2 edit
    ('e',   (2, 1),              0.13), # 2:1 edit
    ('e',   (2, 2),              0.08), # 2:2 edit
#    ('ti',  None,                0.005), # random text insertion
    ('td',  None,                0.02), # random text deletion
    ]

# create discrete distribution for characher mutations
MUT_DIST = stats.rv_discrete(
    name='cmut',
    values=(numpy.arange(len(char_mutations)),
            [i[2] for i in char_mutations]))
    
#------------------------------------------------------------------------------

def basename(fullname):
        '''
        let's assume fullname is always well formed
        '''
        return fullname.split(NAME_SEPARATOR)[0]

#------------------------------------------------------------------------------

def get_texts(indirs, read_files=True):
    '''
    Yields texts for .txt files in a given directory list indirs

    :param indirs: list of directories
    :param read_files: if file contents should be yielded
    '''
    if type(indirs) == str:
        indirs = [indirs]
    for indir in indirs:
        LOG.debug('Processing: {}'.format(indir))
        if not os.path.isdir(indir):
            LOG.debug('Not a directory: {}'.format(indir))
        for filename in os.listdir(indir):
            filepath = os.path.join(indir, filename)
            # skip directories
            if os.path.isdir(filepath):
                continue
            name, ext = os.path.splitext(filename)
            if ext != '.txt':
                LOG.debug('Skipping file: {}'.format(filename))
                continue
            if read_files:
                with open(filepath, 'r') as fin:
                    yield name, fin.read()
            else:
                yield name

#------------------------------------------------------------------------------

#@utils.timeit
def corrupt_ocr(raw_text, p):
    '''
    Introduce p% character OCR errors to given text

    :param raw_text: text to corrupt
    :param p: float, corruption parameter in % (0-100)
    '''

    def gen_rnd_char():
        '''
        Return random char of the english alphabet
        '''
        return random.choice(LETTERS)

    def locate_nearest(text, text_length, pos, char):
        '''
        Locate nearest position for char in text starting.
        '''
        while text[pos] != char:
            pos += 1
            if pos >= text_length:
                return -1
        return pos
    
    #tokens = nltk.word_tokenize(raw_text)
    #lang_model = nltk.Text(tokens)
    text = bytearray(raw_text)
    text_length = len(text)
    num_to_change = int(p/100*text_length)
    num_changed = 0
    mutations = defaultdict(int)
    while num_changed < num_to_change:
        text_length = len(text) # calc new text_length after each iteration
        # generate random mutation
        mut_type, mut_char, mut_prob = char_mutations[MUT_DIST.rvs()]
        if not mut_char:
            # any char; get at random
            mut_char = gen_rnd_char()
        # generate random position to change
        pos = random.randrange(0, text_length - 1)
        # INSERT character
        if mut_type == 'i':
            text.insert(pos, mut_char)
        # DELETE character
        elif mut_type == 'd':
            # if mut_char:
            #     loc = locate_nearest(text, text_length,pos, mut_char)
            #     if loc > -1:
            #         del text[loc]
            # else:
            #     del text[pos]
            del text[pos]
        # EDIT character
        elif mut_type == 'e':
            old_char, new_char = mut_char
            # EDIT specific
            if type(old_char) == type(new_char) == str:
                # edit chat at position
                loc = locate_nearest(text, text_length, pos, old_char)
                if loc >= -1:
                    text[pos] = ord(new_char)
                    num_changed += 1
            # EDIT any
            elif type(old_char) == type(new_char) == int:
                # any char substitution
                subst = [gen_rnd_char() for i in range(new_char)]
                pos_end = pos
                if old_char == 2:
                    pos_end = pos + 2
                text[pos:pos_end] = subst
                num_changed += 1
        # INSERT RANDOM TEXT
        elif mut_type == 'ti':
            pass
            #rnd_text = lang_model.generate(random.randrange(2, 512))
            #text[pos:pos] = rnd_text
        # DELETE RANDOM TEXT
        elif mut_type == 'td':
            del text[pos:pos+random.randrange(1, 128)]
    return str(text)

#------------------------------------------------------------------------------

def corrupt(params):
    '''
    Corrupt text in parallel

    :param params: (output directory, filename, input text, corruption rate)
    '''
    outdir, name, text, p = params
    corrupted_name = '{}{}{}'.format(name, NAME_SEPARATOR, '{:1.3f}'.format(p))
    LOG.info('Writing {} with p={:1.3f}'.format(corrupted_name, p))
    with open(os.path.join(outdir, corrupted_name + '.txt'), 'w') as fout:
        fout.write(corrupt_ocr(text, p))
    return corrupted_name

#------------------------------------------------------------------------------

@utils.timeit
def gen_corrupted_texts(indir, outdir, num_processes=10):
    '''
    Spawn corruption in parallel for every text file in a given input
    directory and write corrupted texts to output directory

    :param indir: input directory
    :param outdir: output directory
    :param num_processes: default number of processes in the process pool
    '''
    # remove existing texts in outdir
    if os.path.isdir(outdir):
        shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir)
    pool = Pool(processes=num_processes)
    # corrupt files in range:
    #corrupt_range = [float(p)/1000 for p in range(5, 450, 50)]
    corrupt_range = [float(random.randrange(0, 4000))/1000 for i in range(9)]
    numd = len(corrupt_range)
    LOG.info('Will generate {} versions of each file'.format(numd))
    for name, text in get_texts(indir):
        # write original file as is
        orig_filename = name + NAME_SEPARATOR
        LOG.info('Writing {}'.format(orig_filename)) 
        with open(os.path.join(outdir, orig_filename + '.txt'), 'w') as fout:
            fout.write(text)
        # write corrupted texts
        # sequential
        #for p in corrupt_range:
        #    corrupt((outdir, name, text, p))
        result = pool.map(corrupt, zip([outdir]*numd,
                                       [name]*numd,
                                       [text]*numd,
                                       corrupt_range))
    pool.close()
    pool.join()

#------------------------------------------------------------------------------

def write_gold_clusters(indir):
    '''
    Write gold cluster pairs to file

    :param indir: list of dirs or single dir
    '''
    gold_filename = 'gold_clusters.pickle'
    # read all book file names
    files = list(get_texts(indir, read_files=False))
    scores = {}
    n = len(files)
    # write cluster membership for all distinct pairs from files (n*(n-1)/2)
    for i in range(n):
        for j in range(n):
            if files[i] != files[j]:
                key = sorted((files[i], files[j]))
                scores[', '.join(key)] = \
                    basename(files[i]) == basename(files[j])
    # write pickled scores to gold_filename
    with open(os.path.join(indir if type(indir) == str else indir[0],
                           gold_filename), 'w') as goldout:
        LOG.info('Writing {}'.format(gold_filename))
        pickle.dump(scores, goldout)

#------------------------------------------------------------------------------

def do_analysis(indir):
    '''
    Extracts some useful info from the generated synthetic dataset

    :param indir: list of dirs or single dir
    '''
    import matplotlib.pyplot as plt
    files = list(get_texts(indir, read_files=False))
    canonical = set([basename(f) for f in files])
    n_canonical = len(canonical)
    print('Num of canonical texts: {}'.format(n_canonical))
    print('Num of texts: {}'.format(len(files)))
    word_count = {c:[] for c in canonical}
    for name, text in get_texts(indir):
        word_count[basename(name)].append(len(text.split()))
    for cname in word_count:
        sizes = sorted(word_count[cname])
        plt.plot(sizes, 'r-') 
    plt.show()

#------------------------------------------------------------------------------
    
if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s : %(message)s',
                        level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',
                        dest='i',
                        help='path to input directory',
                        required=False,
                        nargs='*')
    parser.add_argument('--o',
                        dest='o',
                        help='path to output directory',
                        required=False,
                        type=str)
    parser.add_argument('--p',
                        dest='p',
                        help='percent of character corruption',
                        required=False,
                        type=float)
    parser.add_argument('--gen',
                        help='generate corrupted files',
                        default=False,
                        const=True,
                        nargs='?')
    parser.add_argument('--gold',
                        help='write gold clusters file',
                        default=False,
                        const=True,
                        nargs='?')
    parser.add_argument('--analysis',
                        help='print some dataset insights',
                        default=False,
                        const=True,
                        nargs='?')
    args = parser.parse_args()
    # --gen --i --o [--gold]
    if args.gen and args.i and args.o:
        gen_corrupted_texts(args.i, args.o)
        if args.gold:
            write_gold_clusters(args.o)
    # --gold --i
    elif args.gold and args.i:
        write_gold_clusters(args.i)
    # --analysis --i
    elif args.analysis and args.i:
        do_analysis(args.i)
    # --p
    elif args.p:
        print(corrupt_ocr('test ' * 100, args.p))
    else:
        print('unknown combination of params')
