import sys

from preprocessing.glove import preprocess_glove
from preprocessing.squad_preprocess import preprocess_squad
from preprocessing.trec import preprocess_trec

if __name__ == '__main__':
    targets = sys.argv[1:]
    if not targets:
        print('No targets specified')

    if 'all' in targets:
        targets = ['glove', 'squad', 'trec']

    for target in targets:
        if target == 'glove':
            preprocess_glove()
        elif target == 'squad':
            preprocess_squad()
        elif target == 'trec':
            preprocess_trec()
        else:
            print('Unknown target "{}"'.format(target))
