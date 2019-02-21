# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Preprocess the iwslt 2016 datasets.
'''

import os
import errno
import sentencepiece as spm
import re
from hparams import Hparams
import logging

logging.basicConfig(level=logging.INFO)

def prepro(hp):
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """
    logging.info("# Check if raw files exist")
    train1 = "iwslt2016/de-en/train.tags.de-en.de"
    train2 = "iwslt2016/de-en/train.tags.de-en.en"
    eval1 = "iwslt2016/de-en/IWSLT16.TED.tst2013.de-en.de.xml"
    eval2 = "iwslt2016/de-en/IWSLT16.TED.tst2013.de-en.en.xml"
    test1 = "iwslt2016/de-en/IWSLT16.TED.tst2014.de-en.de.xml"
    test2 = "iwslt2016/de-en/IWSLT16.TED.tst2014.de-en.en.xml"
    for f in (train1, train2, eval1, eval2, test1, test2):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    logging.info("# Preprocessing")
    # train
    _prepro = lambda x:  [line.strip() for line in open(x, 'r').read().split("\n") \
                      if not line.startswith("<")]
    prepro_train1, prepro_train2 = _prepro(train1), _prepro(train2)
    assert len(prepro_train1)==len(prepro_train2), "Check if train source and target files match."

    # eval
    _prepro = lambda x: [re.sub("<[^>]+>", "", line).strip() \
                     for line in open(x, 'r').read().split("\n") \
                     if line.startswith("<seg id")]
    prepro_eval1, prepro_eval2 = _prepro(eval1), _prepro(eval2)
    assert len(prepro_eval1) == len(prepro_eval2), "Check if eval source and target files match."

    # test
    prepro_test1, prepro_test2 = _prepro(test1), _prepro(test2)
    assert len(prepro_test1) == len(prepro_test2), "Check if test source and target files match."

    logging.info("Let's see how preprocessed data look like")
    logging.info("prepro_train1:", prepro_train1[0])
    logging.info("prepro_train2:", prepro_train2[0])
    logging.info("prepro_eval1:", prepro_eval1[0])
    logging.info("prepro_eval2:", prepro_eval2[0])
    logging.info("prepro_test1:", prepro_test1[0])
    logging.info("prepro_test2:", prepro_test2[0])

    logging.info("# write preprocessed files to disk")
    os.makedirs("iwslt2016/prepro", exist_ok=True)
    def _write(sents, fname):
        with open(fname, 'w') as fout:
            fout.write("\n".join(sents))

    _write(prepro_train1, "iwslt2016/prepro/train.de")
    _write(prepro_train2, "iwslt2016/prepro/train.en")
    _write(prepro_train1+prepro_train2, "iwslt2016/prepro/train")
    _write(prepro_eval1, "iwslt2016/prepro/eval.de")
    _write(prepro_eval2, "iwslt2016/prepro/eval.en")
    _write(prepro_test1, "iwslt2016/prepro/test.de")
    _write(prepro_test2, "iwslt2016/prepro/test.en")

    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs("iwslt2016/segmented", exist_ok=True)
    train = '--input=iwslt2016/prepro/train --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=iwslt2016/segmented/bpe --vocab_size={} \
             --model_type=bpe'.format(hp.vocab_size)
    spm.SentencePieceTrainer.Train(train)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("iwslt2016/segmented/bpe.model")

    logging.info("# Segment")
    def _segment_and_write(sents, fname):
        with open(fname, "w") as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train1, "iwslt2016/segmented/train.de.bpe")
    _segment_and_write(prepro_train2, "iwslt2016/segmented/train.en.bpe")
    _segment_and_write(prepro_eval1, "iwslt2016/segmented/eval.de.bpe")
    _segment_and_write(prepro_eval2, "iwslt2016/segmented/eval.en.bpe")
    _segment_and_write(prepro_test1, "iwslt2016/segmented/test.de.bpe")

    logging.info("Let's see how segmented data look like")
    print("train1:", open("iwslt2016/segmented/train.de.bpe",'r').readline())
    print("train2:", open("iwslt2016/segmented/train.en.bpe", 'r').readline())
    print("eval1:", open("iwslt2016/segmented/eval.de.bpe", 'r').readline())
    print("eval2:", open("iwslt2016/segmented/eval.en.bpe", 'r').readline())
    print("test1:", open("iwslt2016/segmented/test.de.bpe", 'r').readline())

if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")