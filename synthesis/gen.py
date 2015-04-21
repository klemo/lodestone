#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------

'''
lodestone (let's hash books)

generates synthetic text dataset by corrupting referent dataset with OCR errors

$ python gen.py --i ref_texts/ --o syn_texts/
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

#------------------------------------------------------------------------------

LOG = logging.getLogger(__name__)

#------------------------------------------------------------------------------

NAME_SEPARATOR = '__'
LETTERS = string.ascii_letters + string.digits

#------------------------------------------------------------------------------
            
def basename(fullname):
        '''
        let's assume fullname is always well formed
        '''
        return fullname.split(NAME_SEPARATOR)[0]

#------------------------------------------------------------------------------

def get_texts(indirs):
    '''
    Yields texts for .txt files in a given directory list indirs
    '''
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
            with open(filepath, 'r') as fin:
                yield name, fin.read()

#------------------------------------------------------------------------------

def corrupt_ocr(raw_text, p):
    '''
    Introduce p% character OCR errors to given text

    :param raw_text: text to corrupt
    :param p: float, corruption parameter 0.0-1.0
    '''

    def gen_rnd_char():
        '''
        Return random char of the english alphabet
        '''
        # TODO: digits should be less probable, not equally
        return random.choice(LETTERS)
    
    # (insert|delete|edit, char, probability)
    # None -> any char
    char_mutations = [
        ('i',  ord(' '),            0.32), # space insert
        ('d',  ord(' '),            0.06), # space deletion
        ('i',  None,                0.04), # any char insert
        ('d',  None,                0.05), # any char delete
        ('e',  (1, 1),              0.25), # 1:1 edit
        ('e',  (1, 2),              0.06), # 1:2 edit
        ('e',  (2, 1),              0.14), # 2:1 edit
        ('e',  (2, 2),              0.08), # 2:2 edit
        ]
    # create discrete distribution for characher mutations
    mut_dist = stats.rv_discrete(
        name='cmut',
        values=(numpy.arange(len(char_mutations)),
                [i[2] for i in char_mutations]))
    #words = text.split()
    text = bytearray(raw_text)
    text_length = len(text)
    num_to_change = int(p*text_length)
    num_changed = 0
    mutations = defaultdict(int)
    while num_changed < num_to_change:
        # generate random mutation
        mut_type, mut_char, mut_prob = char_mutations[mut_dist.rvs()]
        #mutations[(mut_type, mut_char, mut_prob)] += 1
        if not mut_char:
            # any char; get at random
            mut_char = gen_rnd_char()
        # generate random word to change
        pos = random.randrange(0, text_length - 1)
        # INSERT character
        if mut_type == 'i':
            text.insert(pos, mut_char)
        # DELETE character
        elif mut_type == 'd':
            del text[pos]
        # EDIT character
        elif mut_type == 'e':
            old_char, new_char = mut_char
            # EDIT specific
            if type(old_char) == type(new_char) == str:
                # edit chat at position
                # (or locate nearest position for mutation)
                # # (go right, then left: 1, -2, 3, -4...)
                offset, direction = (1, 1)
                while text[pos] != ord(old_char):
                    offset, direction = (offset + 1, direction*(-1))
                    pos = offset * direction
                    if pos < 0 or pos >= text_length:
                        break
                    if 0 <= pos < text_length:
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
    return str(text)

#------------------------------------------------------------------------------

def corrupt(params):
    '''
    Corrupt text in parallel

    :param params: (output directory, filename, input text, corruption rate)
    '''
    outdir, name, text, p = params
    cp = float(p)/1000
    corrupted_name = '{}{}{:03d}'.format(name, NAME_SEPARATOR, p)
    LOG.info('Writing {} with p={:.3f}'.format(corrupted_name, cp))
    with open(os.path.join(outdir, corrupted_name + '.txt'), 'w') as fout:
        fout.write(corrupt_ocr(text, cp))
    return corrupted_name

#------------------------------------------------------------------------------
def gen_corrupted_texts(indir, outdir, num_processes=20):
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
    corrupt_range = [p for p in range(15, 55, 5)]
    numd = len(corrupt_range)
    LOG.info('Will generate {} versions of each file'.format(numd))
    wfiles = []
    for name, text in get_texts(indir):
        # write original file as is
        orig_filename = name + NAME_SEPARATOR
        LOG.info('Writing {}'.format(orig_filename)) 
        with open(os.path.join(outdir, orig_filename + '.txt'), 'w') as fout:
            fout.write(text)
        wfiles.append(orig_filename)
        # write corrupted texts
        result = pool.map(corrupt, zip([outdir]*numd,
                                       [name]*numd,
                                       [text]*numd,
                                       corrupt_range))
        wfiles.extend(result)
    pool.close()
    pool.join()
    # write gold clusters
    scores = {}
    n = len(wfiles)
    for i in range(n):
        for j in range(n):
            if wfiles[i] != wfiles[j]:
                scores[(wfiles[i], wfiles[j])] = \
                    basename(wfiles[i]) == basename(wfiles[j])
    with open(os.path.join(outdir, 'gold_clusters.tmp'), 'w') as goldout:
        print scores
        pickle.dump(scores, goldout)

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
                        default='out',
                        required=False,
                        type=str)
    parser.add_argument('--p',
                        dest='p',
                        help='percent of character corruption',
                        required=False,
                        type=float)
    args = parser.parse_args()
    if args.i:
        gen_corrupted_texts(args.i, args.o)
    elif args.p:
        print(corrupt_ocr('test ' * 100, args.p))
