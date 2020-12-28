# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##set random seed
import numpy as np 
np.random.seed(26)
import tensorflow as tf
tf.set_random_seed(26)

import collections
import csv
import pandas as pd
import os,sys
import modeling
import re
from tensorflow.contrib.layers.python.layers import initializers
# import optimization
# different learning rate for BERT and other parameters
import optimization_layer_lr
import tokenization
import pickle
import codecs
from sklearn import metrics
from sklearn.externals import joblib


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")


## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_word_length", 10,
    "The maximum total input word length"
    "words longer than this will be truncated, and words shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("clean", True, "Whether to clean last training files.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 20, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 20, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
##  different learning rate between BERT's parameters and other parameters
flags.DEFINE_float("other_learning_rate", 1e-5, "The other params learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, start_labels1=None, end_labels1=None, \
                start_labels2=None, end_labels2=None, start_labels3=None, end_labels3=None, \
                start_labels4=None, end_labels4=None, postag_p=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      start_labels: string. The start position label of the entity. This should be
        specified for train and dev examples, but not for test examples.
      end_labels: string. The end position label of the entity. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.start_labels1 = start_labels1
    self.end_labels1 = end_labels1
    self.start_labels2 = start_labels2
    self.end_labels2 = end_labels2
    self.start_labels3 = start_labels3
    self.end_labels3 = end_labels3
    self.start_labels4 = start_labels4
    self.end_labels4 = end_labels4


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               start_labels_ids1,
               end_labels_ids1,
               start_labels_ids2,
               end_labels_ids2,
               start_labels_ids3,
               end_labels_ids3,
               start_labels_ids4,
               end_labels_ids4):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_labels_ids1 = start_labels_ids1
    self.end_labels_ids1 = end_labels_ids1
    self.start_labels_ids2 = start_labels_ids2
    self.end_labels_ids2 = end_labels_ids2
    self.start_labels_ids3 = start_labels_ids3
    self.end_labels_ids3 = end_labels_ids3
    self.start_labels_ids4 = start_labels_ids4
    self.end_labels_ids4 = end_labels_ids4


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class NerProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "merge_train.out")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.out")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.out")), "test")

  def get_labels(self):
    """See base class."""
    labels = ['0','1']
    return labels

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""

    examples = []
    for (i, line) in enumerate(lines):
      # if set_type == 'train':
      #   if i >= int(len(lines)*0.4): ## control dataset
      #     continue
      # if set_type == 'dev':
      #   if i >= int(len(lines)*0.01): ## 
      #     continue
      # if set_type == 'test':
      #   if i >= int(len(lines)*0.01): ## 
      #     continue

      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[0].strip())
      text_b = None
      # four kinds entity type
      start_labels1 = tokenization.convert_to_unicode(line[1].strip())
      end_labels1 = tokenization.convert_to_unicode(line[2].strip())
      start_labels2 = tokenization.convert_to_unicode(line[3].strip())
      end_labels2 = tokenization.convert_to_unicode(line[4].strip())
      start_labels3 = tokenization.convert_to_unicode(line[5].strip())
      end_labels3 = tokenization.convert_to_unicode(line[6].strip())
      start_labels4 = tokenization.convert_to_unicode(line[7].strip())
      end_labels4 = tokenization.convert_to_unicode(line[8].strip())
    

      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, start_labels1=start_labels1,\
                      end_labels1=end_labels1, start_labels2=start_labels2, end_labels2=end_labels2,\
                      start_labels3=start_labels3, end_labels3=end_labels3, start_labels4=start_labels4,\
                      end_labels4=end_labels4))
    return examples


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""


  all_start_labels1 = []
  all_end_labels1 = []
  all_start_labels2 = []
  all_end_labels2 = []
  all_start_labels3 = []
  all_end_labels3 = []
  all_start_labels4 = []
  all_end_labels4 = []


  text_a = example.text_a.split(' ')
  start_labels1 = example.start_labels1.split(' ')
  end_labels1 = example.end_labels1.split(' ')
  start_labels2 = example.start_labels2.split(' ')
  end_labels2 = example.end_labels2.split(' ')
  start_labels3 = example.start_labels3.split(' ')
  end_labels3 = example.end_labels3.split(' ')
  start_labels4 = example.start_labels4.split(' ')
  end_labels4 = example.end_labels4.split(' ')


  # text_a_start_labels = []
  # text_a_end_labels = []
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  all_start_labels1.append(0)
  all_end_labels1.append(0)
  all_start_labels2.append(0)
  all_end_labels2.append(0)
  all_start_labels3.append(0)
  all_end_labels3.append(0)
  all_start_labels4.append(0)
  all_end_labels4.append(0)
  segment_ids.append(0)

  # print('**'*30)
  # print(len(text_a))
  # print(len(text_b))
  # print(len(start_labels))
  # print(len(end_labels))


# process passage sequence
  for i, word in enumerate(text_a):
    # tokenize for each word
    token = tokenizer.tokenize(word)
    tokens.extend(token)
    tmp_s_label1 = start_labels1[i]
    tmp_e_label1 = end_labels1[i]
    tmp_s_label2 = start_labels2[i]
    tmp_e_label2 = end_labels2[i]
    tmp_s_label3 = start_labels3[i]
    tmp_e_label3 = end_labels3[i]
    tmp_s_label4 = start_labels4[i]
    tmp_e_label4 = end_labels4[i]

  
    for m in range(len(token)):
      if m == 0:
        all_start_labels1.append(tmp_s_label1)
        all_end_labels1.append(0)
        all_start_labels2.append(tmp_s_label2)
        all_end_labels2.append(0) 
        all_start_labels3.append(tmp_s_label3)
        all_end_labels3.append(0) 
        all_start_labels4.append(tmp_s_label4)
        all_end_labels4.append(0)
        segment_ids.append(0)
        
      else: # if a word is tokenized
        all_start_labels1.append(0)
        all_end_labels1.append(0)
        all_start_labels2.append(0)
        all_end_labels2.append(0)
        all_start_labels3.append(0)
        all_end_labels3.append(0)
        all_start_labels4.append(0)
        all_end_labels4.append(0)
        segment_ids.append(0)

    # avoid tokenization problem      
    if tmp_e_label1 == '1':
      all_end_labels1[-1] = 1
    if tmp_e_label2 == '1':
      all_end_labels2[-1] = 1
    if tmp_e_label3 == '1':
      all_end_labels3[-1] = 1
    if tmp_e_label4 == '1':
      all_end_labels4[-1] = 1

  tokens.append("[SEP]")
  all_start_labels1.append(0)
  all_end_labels1.append(0)
  all_start_labels2.append(0)
  all_end_labels2.append(0)
  all_start_labels3.append(0)
  all_end_labels3.append(0)
  all_start_labels4.append(0)
  all_end_labels4.append(0)
  segment_ids.append(0)


  
  # process wordshape feature

   
  # 序列截断
  if len(tokens) >= max_seq_length - 1:
    tokens = tokens[:(max_seq_length - 1)] 
    all_start_labels1 = all_start_labels1[:(max_seq_length - 1)]
    all_end_labels1 = all_end_labels1[:(max_seq_length - 1)]
    all_start_labels2 = all_start_labels2[:(max_seq_length - 1)]
    all_end_labels2 = all_end_labels2[:(max_seq_length - 1)]
    all_start_labels3 = all_start_labels3[:(max_seq_length - 1)]
    all_end_labels3 = all_end_labels3[:(max_seq_length - 1)]
    all_start_labels4 = all_start_labels4[:(max_seq_length - 1)]
    all_end_labels4 = all_end_labels4[:(max_seq_length - 1)]
    segment_ids = segment_ids[:(max_seq_length - 1)]
  # print('1:', len(all_postag))
  tokens.append("[SEP]")
  all_start_labels1.append(0)
  all_end_labels1.append(0)
  all_start_labels2.append(0)
  all_end_labels2.append(0)
  all_start_labels3.append(0)
  all_end_labels3.append(0)
  all_start_labels4.append(0)
  all_end_labels4.append(0)
  segment_ids.append(1)



  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)
  # print('2:', len(all_postag))
  # print('22:', len(input_ids))
  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    # we don't concerned about it!
    all_start_labels1.append(0)
    all_end_labels1.append(0)
    all_start_labels2.append(0)
    all_end_labels2.append(0)
    all_start_labels3.append(0)
    all_end_labels3.append(0)
    all_start_labels4.append(0)
    all_end_labels4.append(0)

  
  # print(len(all_postag))
  all_start_labels_ids1 = [label_map[str(i)] for i in all_start_labels1]
  all_end_labels_ids1 = [label_map[str(i)] for i in all_end_labels1]
  all_start_labels_ids2 = [label_map[str(i)] for i in all_start_labels2]
  all_end_labels_ids2 = [label_map[str(i)] for i in all_end_labels2]
  all_start_labels_ids3 = [label_map[str(i)] for i in all_start_labels3]
  all_end_labels_ids3 = [label_map[str(i)] for i in all_end_labels3]
  all_start_labels_ids4 = [label_map[str(i)] for i in all_start_labels4]
  all_end_labels_ids4 = [label_map[str(i)] for i in all_end_labels4]
 

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(all_start_labels_ids1) == max_seq_length
  assert len(all_end_labels_ids1) == max_seq_length
  assert len(all_start_labels_ids2) == max_seq_length
  assert len(all_end_labels_ids2) == max_seq_length
  assert len(all_start_labels_ids3) == max_seq_length
  assert len(all_end_labels_ids3) == max_seq_length
  assert len(all_start_labels_ids4) == max_seq_length
  assert len(all_end_labels_ids4) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("start_labels_ids1: %s" % " ".join([str(x) for x in all_start_labels_ids1]))
    tf.logging.info("end_labels_ids1: %s" % " ".join([str(x) for x in all_end_labels_ids1]))
    tf.logging.info("start_labels_ids2: %s" % " ".join([str(x) for x in all_start_labels_ids2]))
    tf.logging.info("end_labels_ids2: %s" % " ".join([str(x) for x in all_end_labels_ids2]))
    tf.logging.info("start_labels_ids3: %s" % " ".join([str(x) for x in all_start_labels_ids3]))
    tf.logging.info("end_labels_ids3: %s" % " ".join([str(x) for x in all_end_labels_ids3]))
    tf.logging.info("start_labels_ids4: %s" % " ".join([str(x) for x in all_start_labels_ids4]))
    tf.logging.info("end_labels_ids4: %s" % " ".join([str(x) for x in all_end_labels_ids4]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      start_labels_ids1=all_start_labels_ids1,
      end_labels_ids1=all_end_labels_ids1,
      start_labels_ids2=all_start_labels_ids2,
      end_labels_ids2=all_end_labels_ids2,
      start_labels_ids3=all_start_labels_ids3,
      end_labels_ids3=all_end_labels_ids3,
      start_labels_ids4=all_start_labels_ids4,
      end_labels_ids4=all_end_labels_ids4
      )
  return feature


def file_based_convert_examples_to_features(
    examples, label_map, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_map,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["start_labels_ids1"] = create_int_feature(feature.start_labels_ids1)
    features["end_labels_ids1"] = create_int_feature(feature.end_labels_ids1)
    features["start_labels_ids2"] = create_int_feature(feature.start_labels_ids2)
    features["end_labels_ids2"] = create_int_feature(feature.end_labels_ids2)
    features["start_labels_ids3"] = create_int_feature(feature.start_labels_ids3)
    features["end_labels_ids3"] = create_int_feature(feature.end_labels_ids3)
    features["start_labels_ids4"] = create_int_feature(feature.start_labels_ids4)
    features["end_labels_ids4"] = create_int_feature(feature.end_labels_ids4)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "start_labels_ids1": tf.FixedLenFeature([seq_length], tf.int64),
      "end_labels_ids1": tf.FixedLenFeature([seq_length], tf.int64),
      "start_labels_ids2": tf.FixedLenFeature([seq_length], tf.int64),
      "end_labels_ids2": tf.FixedLenFeature([seq_length], tf.int64),
      "start_labels_ids3": tf.FixedLenFeature([seq_length], tf.int64),
      "end_labels_ids3": tf.FixedLenFeature([seq_length], tf.int64),
      "start_labels_ids4": tf.FixedLenFeature([seq_length], tf.int64),
      "end_labels_ids4": tf.FixedLenFeature([seq_length], tf.int64)
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:

      # d = d.repeat(1)
      d = d.shuffle(buffer_size=500)

    d = d.apply(
        tf.contrib.data.map_and_batch(
          lambda record: _decode_record(record, name_to_features),
          batch_size=batch_size,
          drop_remainder=drop_remainder))
    return d

  return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,\
                 start_labels_ids, end_labels_ids, num_labels, tag_information, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_sequence_output()
  hidden_size = output_layer.shape[-1].value

  # used = tf.sign(tf.abs(input_ids))
  # lengths = tf.reduce_sum(used, reduction_indices=1)
  # print('length:', lengths.shape)
  
  tag_type_number = 4 #the number of entity type
  
  # if is_training:
  #   output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
  # hidden = tf.reshape(output_layer, shape=[-1, hidden_size])

  if FLAGS.do_train:
        batch_size = FLAGS.train_batch_size
  elif FLAGS.do_eval:
    batch_size = FLAGS.eval_batch_size
  else:
    batch_size = FLAGS.eval_batch_size

# append different entity type information
  tag_information = tf.tile(tf.expand_dims(tag_information, 1), [1, FLAGS.max_seq_length, 1])
  tag_information = tf.tile(tf.expand_dims(tag_information, 1), [1, batch_size, 1, 1])
  tag_information = tf.cast(tag_information, dtype=tf.float32)
  
  print(tag_information.shape)
  hidden_1 = tf.concat([output_layer, tag_information[0]], axis=-1)
  hidden_2 = tf.concat([output_layer, tag_information[1]], axis=-1)
  hidden_3 = tf.concat([output_layer, tag_information[2]], axis=-1)
  hidden_4 = tf.concat([output_layer, tag_information[3]], axis=-1)

  if is_training:
      # output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
      hidden_1 = tf.nn.dropout(hidden_1, keep_prob=0.9)
      hidden_2 = tf.nn.dropout(hidden_2, keep_prob=0.9)
      hidden_3 = tf.nn.dropout(hidden_3, keep_prob=0.9)
      hidden_4 = tf.nn.dropout(hidden_4, keep_prob=0.9)
  

  # hidden_size = output_layer.shape[-1].value
  # hidden = tf.reshape(output_layer, shape=[-1, hidden_size])

  hidden_size = hidden_4.shape[-1].value
  hidden1 = tf.reshape(hidden_1, shape=[-1, hidden_size])
  hidden2 = tf.reshape(hidden_2, shape=[-1, hidden_size])
  hidden3 = tf.reshape(hidden_3, shape=[-1, hidden_size])
  hidden4 = tf.reshape(hidden_4, shape=[-1, hidden_size])

  hiddens = [hidden1, hidden2, hidden3, hidden4]

  start_pred = []
  end_pred = []
  #start position labels logits
  with tf.variable_scope('start_logits'):
    for i in range(tag_type_number):
      start_W = tf.get_variable("start_W_"+str(i), shape=[hidden_size, num_labels],
                        dtype=tf.float32, initializer=initializers.xavier_initializer())
      start_b = tf.get_variable("start_b_"+str(i), shape=[num_labels],
                        dtype=tf.float32, initializer=tf.zeros_initializer())
      start_pred.append(tf.nn.xw_plus_b(hiddens[i], start_W, start_b))

  #end position labels logits
  with tf.variable_scope('end_logits'):
    for i in range(tag_type_number):
      end_W = tf.get_variable("end_W_"+str(i), shape=[hidden_size, num_labels],
                        dtype=tf.float32, initializer=initializers.xavier_initializer())
      end_b = tf.get_variable("end_b_"+str(i), shape=[num_labels],
                        dtype=tf.float32, initializer=tf.zeros_initializer())
      end_pred.append(tf.nn.xw_plus_b(hiddens[i], end_W, end_b))

  total_loss = 0.0
  start_pred_ids = []
  end_pred_ids = []
  with tf.variable_scope("start_loss"):
    for i in range(tag_type_number):
      logits_start = tf.reshape(start_pred[i], [-1, FLAGS.max_seq_length, num_labels])
      log_probs_start = tf.nn.log_softmax(logits_start, axis=-1)
      one_hot_labels_start = tf.one_hot(start_labels_ids[i], depth=num_labels, dtype=tf.float32)
      per_example_loss_start = -tf.reduce_sum(one_hot_labels_start * log_probs_start, axis=-1)
      start_loss = tf.reduce_mean(per_example_loss_start)
      probabilities_start = tf.nn.softmax(logits_start, axis=-1)
      start_pred_ids.append(tf.argmax(probabilities_start,axis=-1))
      
      total_loss += start_loss
 
  with tf.variable_scope("end_start_loss"):
    for i in range(tag_type_number):
      logits_end = tf.reshape(end_pred[i], [-1, FLAGS.max_seq_length, num_labels])
      log_probs_end = tf.nn.log_softmax(logits_end, axis=-1)
      one_hot_labels_end = tf.one_hot(end_labels_ids[i], depth=num_labels, dtype=tf.float32)
      per_example_loss_end = -tf.reduce_sum(one_hot_labels_end * log_probs_end, axis=-1)
      end_loss = tf.reduce_mean(per_example_loss_end)
      probabilities_end = tf.nn.softmax(logits_end, axis=-1)
      end_pred_ids.append(tf.argmax(probabilities_end,axis=-1))

      total_loss += end_loss


  ### BERT
  # total_loss = start_loss + end_loss
  
  return (total_loss, start_pred_ids, end_pred_ids)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, other_learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    start_labels_ids1 = features["start_labels_ids1"]
    end_labels_ids1 = features["end_labels_ids1"]
    start_labels_ids2 = features["start_labels_ids2"]
    end_labels_ids2 = features["end_labels_ids2"]
    start_labels_ids3 = features["start_labels_ids3"]
    end_labels_ids3 = features["end_labels_ids3"]
    start_labels_ids4 = features["start_labels_ids4"]
    end_labels_ids4 = features["end_labels_ids4"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    ## this is entity definition representation, we have defined in advance
    norm_entity_type = np.load('../data/normalize.npy')
    no_norm_entity_type = np.load('../data/non_normalize.npy')
    protein_entity_type = np.load('../data/protein.npy')
    unclear_entity_type = np.load('../data/unclear.npy')
    # norm_entity_type = np.load('../data/normalize_naive_bert.npy')
    # no_norm_entity_type = np.load('../data/non_normalize_naive_bert.npy')
    # protein_entity_type = np.load('../data/protein_naive_bert.npy')
    # unclear_entity_type = np.load('../data/unclear_naive_bert.npy')
    tag_indformation = [norm_entity_type, no_norm_entity_type, protein_entity_type, unclear_entity_type]
    tag_indformation = np.array(tag_indformation)

    start_labels_ids = [start_labels_ids1, start_labels_ids2, start_labels_ids3, start_labels_ids4]
    end_labels_ids = [end_labels_ids1, end_labels_ids2, end_labels_ids3, end_labels_ids4]


    (total_loss, start_pred_ids, end_pred_ids) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, start_labels_ids, end_labels_ids, \
         num_labels, tag_indformation, use_one_hot_embeddings)
    
    pre_1 = tf.stack([start_pred_ids[0],end_pred_ids[0]],axis=1)
    pre_2 = tf.stack([start_pred_ids[1],end_pred_ids[1]],axis=1)
    pre_3 = tf.stack([start_pred_ids[2],end_pred_ids[2]],axis=1)
    pre_4 = tf.stack([start_pred_ids[3],end_pred_ids[3]],axis=1)

    pred_ids= tf.stack([pre_1, pre_2, pre_3, pre_4], axis=1)

    print('-*'*30)
    print(pred_ids)
    
    tvars = tf.trainable_variables()
    scaffold_fn = None
    # 加载BERT模型
    if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                    init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    '''
    tf.logging.info("**** Trainable Variables ****")

    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    '''

    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    # total_parameters = 0 
    # #iterating over all variables 
    # for variable in tf.trainable_variables():   
    #   local_parameters=1 
    #   shape = variable.get_shape()  #getting shape of a variable 
    #   for i in shape: 
    #     local_parameters*=i.value  #mutiplying dimension values 
    #     total_parameters+=local_parameters 
    # print(total_parameters)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization_layer_lr.create_optimizer(
            total_loss, learning_rate, other_learning_rate, num_train_steps, num_warmup_steps, use_tpu)
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)  
    elif mode == tf.estimator.ModeKeys.EVAL:

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            scaffold_fn=scaffold_fn)  #
    else:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=pred_ids,
            scaffold_fn=scaffold_fn
        )
    return output_spec

  return model_fn

def labeltoid(label_list):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
        pickle.dump(label_map, w)

    return label_map


def get_pred_metric(result, eval_input_ids, tokenizer, tag_type_number):
  all_pred_ent = []
  # print(len(result))
  # print(len(eval_input_ids))
  # print(result)
  for i in range(len(result)):
    # print(i)
    for j in range(tag_type_number):
      tmp_input_ids = eval_input_ids[i]
      start_preds = result[i][j][0]
      end_preds = result[i][j][1]
      start_inds = []
      end_inds = []
    # print(start_preds)
    # print(end_preds)
      for ind in range(len(start_preds)):
        if(start_preds[ind]==1):
          start_inds.append(ind) 

      for ind in range(len(end_preds)):
        if(end_preds[ind]==1):
          end_inds.append(ind) 

      if(len(start_inds)==0):
        all_pred_ent.append('')
      else:
        ans = []

        def back(start_inds, end_inds):
            # global ans
            if(len(start_inds)==0 or len(end_inds)==0):
                return 
            while(len(end_inds)>0 and end_inds[0]<start_inds[0]):
                end_inds = end_inds[1:]     
            if(len(end_inds)>0):
                while(len(start_inds)>1 and (end_inds[0]-start_inds[1])>0 and ((end_inds[0]-start_inds[0])>(end_inds[0]-start_inds[1]))):
                    start_inds = start_inds[1:]
                ans.append((start_inds[0],end_inds[0]))
            back(start_inds[1:],end_inds[1:])
        back(start_inds, end_inds)
        if(len(ans)==0):
          all_pred_ent.append('')
        else:
          all_tmp_ent = []
          for item in ans:
            s_ind = item[0]
            e_ind = item[1]
            # print(s_ind, e_ind)
            tmp_ent = ' '.join(tokenizer.convert_ids_to_tokens(tmp_input_ids[s_ind:e_ind+1])).replace(' ##','')
            end_str = ''
            e_ind += 1
            while((e_ind<len(tmp_input_ids)-1) and ('##' in tokenizer.convert_ids_to_tokens([tmp_input_ids[e_ind]])[0])):
              end_str += tokenizer.convert_ids_to_tokens([tmp_input_ids[e_ind]])[0].replace('##','')
              e_ind += 1   
            tmp_ent += end_str
            all_tmp_ent.append(tmp_ent)
            # print(all_tmp_ent)
          all_pred_ent.append(all_tmp_ent)

        # print(' '.join(tokenizer.convert_ids_to_tokens(tmp_input_ids)))
        # print(all_tmp_ent)
  # print(all_pred_ent)
  # print(len(all_pred_ent))

  ## save result in file
  with open(os.path.join(FLAGS.output_dir, 'test_pred_answer.txt'), 'w') as f:
    for entities in all_pred_ent:
      if len(entities) == 0:
        f.write('\n')
      else:
        f.write('\t'.join(entities) + '\n')

  with open(os.path.join(FLAGS.data_dir, 'test_entities.txt'), 'r') as f:
    gold = f.readlines()

  all_pred = 0
  for item in all_pred_ent:
    if(item==''):
      continue 
    else:
      for i in item:
        all_pred += 1

  tp = 0
  all_ann = 0
  for i in range(len(gold)): 
    if(len(gold[i].strip())!=0):
      # print(gold[i])
      for k in gold[i].strip().split('\t'):
        all_ann += 1
  for i in range(len(gold)):
      if(all_pred_ent[i]!=''):
        for j in all_pred_ent[i]:
          for e in gold[i].strip().split('\t'):
            if j.lower() == e.lower():
              tp += 1
              break
  p = tp/all_pred
  r = tp/all_ann
  f = (2*p*r)/(p+r)
  f1 = f
  print(tp,all_pred,all_ann)
  print(p,r,f)
  # print(all_pred_ent)

  return f1



def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "ner": NerProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))
  ## del last training file  
  if(FLAGS.do_train and FLAGS.clean):     
      if os.path.exists(FLAGS.output_dir):
          def del_file(path):
              ls = os.listdir(path)
              for i in ls:
                  c_path = os.path.join(path, i)
                  if os.path.isdir(c_path):
                      del_file(c_path)
                  else:
                      os.remove(c_path)

          try:
              del_file(FLAGS.output_dir)
          except Exception as e:
              print(e)
              print('pleace remove the files of output dir and data.conf')
              exit(-1)


  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()
  label_map = labeltoid(label_list)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  print(tokenizer.convert_ids_to_tokens([101, 2424, 1996, 15316, 4668, 1997, 5423, 15660, 102 ]))
  # sys.exit(0)
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=None,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      other_learning_rate=FLAGS.other_learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      model_dir=FLAGS.output_dir,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_map, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    eval_input_ids = []
    for (ex_index, example) in enumerate(eval_examples):
      feature = convert_single_example(ex_index, example, label_map,
                                     FLAGS.max_seq_length, tokenizer)
      eval_input_ids.append(feature.input_ids)

    num_actual_eval_examples = len(eval_examples)
    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_map, FLAGS.max_seq_length, tokenizer, eval_file)
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

  ## Get id2label
  with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
      label2id = pickle.load(rf)
      id2label = {value: key for key, value in label2id.items()}

  best_result = 0
  all_results = []
  if FLAGS.do_train:
    for i in range(int(FLAGS.num_train_epochs)):
      print('**'*40)
      print('Train {} epoch'.format(i+1))
      estimator.train(input_fn=train_input_fn)

  if FLAGS.do_predict:
    print('***********************Running Prediction************************')
    # print('Use model which perform best on dev data')
    cur_ckpt_path = estimator.latest_checkpoint()

    print('Use model which restore from last ckpt')
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        model_dir=None,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        warm_start_from=cur_ckpt_path)
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_map,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    result = estimator.predict(input_fn=predict_input_fn)
    result = list(result)
    tag_type_number = 4
    print(get_pred_metric(result, eval_input_ids, tokenizer, tag_type_number))

  

if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
  
