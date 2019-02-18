# eval2 = "iwslt2016/segmented/eval.de.bpe"
# eval3 = "iwslt2016/prepro/eval.en"
#
# lines2 = open(eval2, "r").read().splitlines()
# lines3 = open(eval3, "r").read().splitlines()
#
# for l2, l3 in zip(lines2, lines3):
#     print(l2)
#     print(l3)
#     print()



# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file(file_name='logdir/2/iwslt2016_E01L6.23A0.18-1534', tensor_name='', all_tensors=False)
# from utils import  calc_bleu
# calc_bleu("tes", "tes")


# c = "perl multi-bleu.perl iwslt2016/prepro/eval.en < eval/iwslt2016_E05L1.75"
# import os
# os.system(c)
# import math
# print(30680/20)
# import os

# with open("'result-logdir4iwslt2016_E01L1.55A0.98-6135", "w") as f:
#     f.write("a")
# import tensorflow as tf
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     ckpt = tf.train.latest_checkpoint("logdir/2/iwslt2016_E03L1.46A1.00-4602")
#
#     saver.restore(sess, ckpt); print("checkpoint restored from {}".format(ckpt))

# print(a)
# import os
# print(os.path.isdir("logdir/2/iwslt2016_E01L6.23A0.18-1534"))
# import logging
# a=0
# def w():
#     if a == 3:
#         print(1)
#     elif a==5:
#         print(2)
#     else:
#         raise "fff"
#
# a
# logging.basicConfig(level=logging.INFO)
# logging.info('Some information')
# from tensorflow.python import pywrap_tensorflow
# varlist=[]
# reader = pywrap_tensorflow.NewCheckpointReader("logdir/21/my_model_L8.13_A0.03-300")
# var_to_shape_map = reader.get_variable_to_shape_map()
# vars = [v for v in var_to_shape_map if "Adam" not in v]
# print(vars)
# import tensorflow as tf
#
# print( tf.train.latest_checkpoint("logdir/3"))

# import argparse
# from data_load import load_vocab
#
# class Hparams:
#     def __init__(self):
#         self.parser = argparse.ArgumentParser(add_help=False)
#
#     def prepo(self):
#         # raw data file paths (reading)
#         self.parser.add_argument('--raw_train1', default='corpora/train.tags.de-en.de',
#                                  help="german training raw data")
#         self.parser.add_argument('--raw_train2', default='corpora/train.tags.de-en.en',
#                                  help="english training raw data")
#         self.parser.add_argument('--raw_eval1', default='corpora/IWSLT16.TED.tst2013.de-en.de.xml',
#                                  help="german evaluation raw data")
#         self.parser.add_argument('--raw_eval2', default='corpora/IWSLT16.TED.tst2013.de-en.en.xml',
#                                  help="english evaluation raw data")
#         self.parser.add_argument('--raw_test1', default='corpora/IWSLT16.TED.tst2014.de-en.de.xml',
#                                  help="german test raw data")
#         self.parser.add_argument('--raw_test2', default='corpora/IWSLT16.TED.tst2014.de-en.en.xml',
#                                  help="english test raw data")
#
#         # preprocessing (writing)
#         self.parser.add_argument('--prepro', default='prepro',
#                                  help="data path preprocessed data are saved to")
#         self.parser.add_argument('--prepro_train1', default='prepro/train.de',
#                                  help="german training prepo data")
#         self.parser.add_argument('--prepro_train2', default='prepro/train.en',
#                                  help="english training prepo data")
#         self.parser.add_argument('--prepro_train', default='prepro/train',
#                                  help="merged training prepo data")
#         self.parser.add_argument('--prepro_eval1', default='prepro/eval.de',
#                                  help="german evaluation prepo data")
#         self.parser.add_argument('--prepro_eval2', default='prepro/eval.en',
#                                  help="english evaluation prepo data")
#         self.parser.add_argument('--prepro_test1', default='prepro/test.de',
#                                  help="german test prepo data")
#         self.parser.add_argument('--prepro_test2', default='prepro/test.en',
#                                  help="english test prepo data")
#
#         # sentencepiece
#         self.parser.add_argument('--segmented', default='segmented',
#                                  help="data path segmented data are saved to")
#         self.parser.add_argument('--vocab_size', default=32000, type=int)
#         self.parser.add_argument('--pad_id', default=0, type=int)
#         self.parser.add_argument('--unk_id', default=1, type=int)
#         self.parser.add_argument('--bos_id', default=2, type=int)
#         self.parser.add_argument('--eos_id', default=3, type=int)
#         self.parser.add_argument('--model_prefix', default='segmented/bpe')
#         self.parser.add_argument('--model_type', default='bpe')
#
#         # segment
#         self.parser.add_argument('--train_fpath1', default='segmented/train.de.bpe',
#                                  help="german training segmented data")
#         self.parser.add_argument('--train_fpath2', default='segmented/train.en.bpe',
#                                  help="english training segmented data")
#         self.parser.add_argument('--eval_fpath1', default='segmented/eval.de.bpe',
#                                  help="german evaluation segmented data")
#         self.parser.add_argument('--eval_fpath2', default='segmented/eval.en.bpe',
#                                  help="english evaluation segmented data")
#         self.parser.add_argument('--test_fpath1', default='segmented/test.de.bpe',
#                                  help="german test segmented data")
#
#         # self.hp = self.parser.parse_args()
#         # self.hp.test_fpath2 = self.hp.prepro_test2
#
#     def train(self):
#         self.prepo()
#         # self.parser = argparse.ArgumentParser(parents=[self.parser])
#
#         # vocabulary
#         self.parser.add_argument('--vocab_fpath', default='segmented/bpe.vocab')
#
#         # Training
#         self.parser.add_argument('--batch_size', default=128, type=int)
#         self.parser.add_argument('--eval_batch_size', default=128, type=int)
#
#         self.parser.add_argument('--lr', default=0.003, type=int, help="learning rate")
#         self.parser.add_argument('--warmup_steps', default=4000, type=int)
#         self.parser.add_argument('--logdir', default="logdir", help="log directory")
#         self.parser.add_argument('--num_epochs', default=20, type=int)
#         self.parser.add_argument('--eval_steps', default=100, type=int)
#
#         # Model
#         self.parser.add_argument('--d_model', default=512, type=int,
#                             help="hidden dimension of encoder/decoder")
#         self.parser.add_argument('--d_ff', default=2048, type=int,
#                             help="hidden dimension of feedforward layer")
#         self.parser.add_argument('--num_blocks', default=6, type=int,
#                             help="number of encoder/decoder blocks")
#         self.parser.add_argument('--num_heads', default=8, type=int,
#                             help="number of attention heads")
#         self.parser.add_argument('--maxlen1', default=100, type=int,
#                             help="maximum length of a source sequence")
#         self.parser.add_argument('--maxlen2', default=100, type=int,
#                             help="maximum length of a target sequence")
#         self.parser.add_argument('--dropout_rate', default=0.1, type=float)
#         self.parser.add_argument('--smoothing', default=0.1, type=float,
#                             help="label smoothing rate")
#
#         # self.hp = self.parser.parse_args()
#
#         # token2idx, idx2token = load_vocab(self.hp.vocab_fpath)
#         # self.hp.token2idx = token2idx
#         # self.hp.idx2token = idx2token
#
#     def infer(self):
#         self.train()
#         self.parser = argparse.ArgumentParser(parents=[self.parser], add_help=False)
#         self.parser.add_argument('--test_batch_size', default=128, type=int)
#         self.parser.add_argument('--result', default="result", help="result fpath")
#         # self.hp = self.parser.parse_args()