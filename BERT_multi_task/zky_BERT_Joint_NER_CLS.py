#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
# Copyright 2018 The Google AI Language Team Authors.
# Copyright 2019 The BioNLP-HZAU Kaiyin Zhou
# Time:2019/04/08
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pickle
from absl import flags,logging
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import metrics
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,confusion_matrix

FLAGS = flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES']='0'

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

# if you download cased checkpoint you should use "False",if uncased you should use
# "True"
# if we used in bio-medical field，don't do lower case would be better!

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("filter_tag", False, "Filter some unnecessary targ.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 500,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("middle_output", "middle_data", "Dir was used to store middle data!")
flags.DEFINE_string("disease", "BLCA", "disease!")

flags.DEFINE_bool("crf", "False", "use crf!")

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text,filename,class_label, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label
    self.filename=filename
    self.class_label = class_label

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
               mask,
               segment_ids,
               class_labels,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.mask = mask
    self.segment_ids = segment_ids
    self.class_labels = class_labels
    self.label_ids = label_ids
    self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls,input_file):
        """Read a BIO data!"""
        rf = open(input_file,'r')
        lines = [];words = [];labels = []
        all_labels = ["[PAD]",'B-Gene','B-Disease','B-NegReg','B-PosReg','B-Var','B-Reg',
        'I-Disease','I-Gene','I-NegReg','I-PosReg','I-Var','I-Reg','O',"X","[CLS]","[SEP]"]
        # all_labels = ["[PAD]",'B-Gene','B-NegReg','B-PosReg','B-Var','B-Reg','I-Gene','I-NegReg','I-PosReg','I-Var','I-Reg','O',"X","[CLS]","[SEP]"]
        filename=None
        class_label=None
        for line in rf:
            if len(line.strip())!=0:
                word = line.rstrip().split('\t')[0]
                label = line.rstrip().split('\t')[-1]
                if FLAGS.filter_tag:
                    if label not in all_labels:
                        label = "O"
                class_label = line.rstrip().split('\t')[2]
                filename = line.rstrip().split('\t')[1]
                words.append(word)
                labels.append(label)
            # here we dont do "DOCSTART" check
            else:
                #get class label
                l = ' '.join([label for label in labels])
                w = ' '.join([word for word in words])
                lines.append((l, filename, class_label, w))
                words=[]
                labels = []
        rf.close()
        return lines

class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "{}.txt".format(FLAGS.disease))), "dev"
        )

    def get_test_examples(self,data_dir):
        print(os.path.join(data_dir, "{}.txt".format(FLAGS.disease)))
        return self._create_example(
            self._read_data(os.path.join(data_dir, "{}.txt".format(FLAGS.disease))), "test"
        )


    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        #return ["[PAD]","B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]
        if FLAGS.filter_tag:
            return ["[PAD]",'B-Gene','B-Disease','B-NegReg','B-PosReg','B-Var','B-Reg','I-Disease','I-Gene','I-NegReg','I-PosReg','I-Var','I-Reg','O',"X","[CLS]","[SEP]"]
        else:
            return ["[PAD]",'B-Gene','B-Disease','B-Enzyme','B-CPA','B-MPA','B-NegReg','B-Pathway','B-PosReg','B-Protein','B-Reg','B-Var','B-Interaction',
            'I-Disease','I-Interaction','I-Reg','I-CPA','I-Gene','I-NegReg','I-Enzyme','I-MPA','I-Pathway','I-PosReg','I-Protein','I-Var','O',"X","[CLS]",
            "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[3])
            class_labels = tokenization.convert_to_unicode(line[2])
            filenames = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts,filename=filenames,class_label=class_labels, label=labels))
        return examples

def labelmap():
    x = {1:[1,0], 2:[0,1], 3:[1,1], 4:[0,0]}
    return x
def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature
    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]
    """
    hasmap = labelmap()
    label_map = {}
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i
    with open(FLAGS.middle_output+"/label2id.pkl",'wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    class_label = example.class_label
    filename = example.filename
    tokens = []
    labels = []
    class_labels = []
    filenames = []
    class_labels.extend(hasmap[int(class_label)])
    filenames.append(filename)
    true_token = []
    for i,(word,label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                true_token.append(word)
                labels.append(label)
            else:
                labels.append("X")
                true_token.append("**###**")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
        true_token = true_token[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntrue_tokens=[]
    ntrue_tokens.append("[CLS]")
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
        ntrue_tokens.append(true_token[i])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1]*len(input_ids)
    #use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
        ntrue_tokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    assert len(ntrue_tokens) == max_seq_length
    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        logging.info("ntrue_tokens: %s" % " ".join([str(x) for x in ntrue_tokens]))
        logging.info("ntokens: %s" % " ".join([str(x) for x in ntokens]))

    # TODO  class_labels
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        class_labels = class_labels,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature, ntokens,filenames,ntrue_tokens

def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file,mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
 
    batch_filename=[]
    batch_true_token=[]
    single_class_label=[]
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature, ntokens, filenames ,true_token = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        batch_tokens.extend(ntokens)
        batch_labels.extend(feature.label_ids)

        single_class_label.extend(feature.class_labels)
        batch_true_token.extend(true_token)
        batch_filename.extend(filenames*len(ntokens))
        def create_int_feature(values,use_class=False):
            if use_class:
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
                return f
            else:
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["mask"] = create_int_feature(feature.mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["class_labels"] = create_int_feature(feature.class_labels,True)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()
    return batch_tokens,batch_labels,single_class_label,batch_filename,batch_true_token

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "class_labels": tf.FixedLenFeature([2], tf.int64),

    }
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn

# all above are related to data preprocess
# Following i about the model

def hidden2tag(hiddenlayer,numclass):
    linear = tf.keras.layers.Dense(numclass,activation=None)
    return linear(hiddenlayer)

def crf_loss(logits,labels,mask,num_labels,mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    #TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
                "transition",
                shape=[num_labels,num_labels],
                initializer=tf.contrib.layers.xavier_initializer()
        )
    
    log_likelihood,transition = tf.contrib.crf.crf_log_likelihood(logits,labels,transition_params=trans ,sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)
   
    return loss,transition

def softmax_layer(logits,labels,num_labels,mask):
    reshape_logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])


    mask = tf.cast(mask,dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=reshape_logits,onehot_labels=one_hot_labels)
    # loss *= tf.reshape(mask, [-1])
    # loss = tf.reduce_sum(loss)
    # total_size = tf.reduce_sum(mask)
    # total_size += 1e-12 # to avoid division by 0 for all-0 weights
    # loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    predict = tf.reshape(predict,shape = [-1,FLAGS.max_seq_length])
    # print(probabilities)
    # predprob = tf.math.reduce_max(probabilities)
    # predprob = tf.math.argmax(probabilities, axis=-1)
    return loss, predict

# use context vector ?Hierarchical Attention Networks for Document Classification?
# word level attention
class Attention(tf.keras.layers.Layer):
    def __init__(self,hidden_size,attention_size):
        super(Attention, self).__init__()
        self.W = tf.get_variable("W",shape=[hidden_size,attention_size],initializer=tf.glorot_normal_initializer())
        self.B = tf.get_variable("B",shape=[attention_size],initializer=tf.glorot_normal_initializer())
        self.U= tf.get_variable("U",shape=[attention_size],initializer=tf.glorot_normal_initializer())

    def __call__(self,encoder_output):#[batch,sequence_len,feats_dim]
        U= tf.math.tanh(tf.tensordot(encoder_output,self.W,axes=1)+self.B)#[batch,sequence_len, attention_size]
        A = tf.tensordot(U,self.U,axes=1)#[batch,sequence_len]
        alphas = tf.keras.backend.softmax(A)#[batch,sequence_len]
        output = tf.math.reduce_sum(encoder_output*tf.expand_dims(alphas,-1),1)#[batch,feats_dim]   
        return output, alphas

def attention_loss(output_layer,hidden_size,num_labels,class_label,mask):

    """
    attention_output:[B]

    class_label:could be 0 or 1; 0 means don't need tag and 1 means need tag.
    
    word level attention loss, help do classfication!

    """
    def attention2dim(attention_output,hidden_size):
        linear = tf.keras.layers.Dense(hidden_size,activation="relu")
        return linear(attention_output)
    
    attention_output, attention_weights = Attention(hidden_size,num_labels)(output_layer[:,1:,:])
    output = attention2dim(attention_output,hidden_size)#[B*C]
    ##############different ways to use attention results!##############
    attention_result = tf.concat([output_layer[:,0,:], output], 1)#[B*2C]
    # attention_result = output
    # attention_result = output_layer[:,0,:]*output
    # attention_result = tf.add(output_layer[:,0,:], output)
    # attention_result = output_layer[:,0,:]
    ##############different ways to use attention results!##############
    class_result = tf.keras.layers.Dense(2,activation=None)(attention_result)#[B*5]
    attention_loss_ = tf.losses.sigmoid_cross_entropy(logits=class_result,multi_class_labels=class_label)
    # attention_loss_ *= tf.cast(tf.reshape(mask, [-1]),tf.float32)
    # attention_loss_ = tf.reduce_sum(attention_loss_)
    # total_size = tf.reduce_sum(mask)
    # total_size += 1e-12 # to avoid division by 0 for all-0 weights
    # attention_loss_ /= total_size
    prob_label = tf.math.sigmoid(class_result)
    true_label = class_label# True label
    return attention_loss_,attention_weights,true_label,prob_label


def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, class_label, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config = bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
        )

    output_layer = model.get_sequence_output()
    #output_layer shape is
    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.2)(output_layer)
    logits = hidden2tag(output_layer,num_labels)
    # TODO test shape
    logits = tf.reshape(logits,[-1,FLAGS.max_seq_length,num_labels])
    # if FLAGS.crf:
    if FLAGS.crf:
        mask2len = tf.reduce_sum(mask,axis=1)
        loss, trans = crf_loss(logits,labels,mask,num_labels,mask2len)
        predict,viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        probabilities = tf.math.softmax(logits, axis=-1)
        # predprob = tf.reduce_max(probabilities,axis=-1)#?crf_decode
        logits = tf.concat([logits,probabilities],axis=-1)
        attention_loss_ ,attention_weights, true_label,prob_label = attention_loss(logits,logits.shape[-1],num_labels,class_label,mask)
    else:
        loss,predict  = softmax_layer(logits, labels, num_labels, mask)

        probabilities = tf.math.softmax(logits, axis=-1)
        logits = tf.concat([logits,probabilities],axis=-1)
        # predprob = tf.zeros_like(predict)
        attention_loss_ ,attention_weights, true_label,prob_label = attention_loss(logits,logits.shape[-1],num_labels,class_label,mask)
    # Here we add attention loss 
    # [CLS] result plus attention result
    # here we let attention_size=num_labels hidden_size=output_layer dim
    # gamma = tf.Variable(initial_value=1,trainable=True)
    # return (gamma*loss + attention_loss_, logits,predict,attention_weights,true_label,prob_label)
    return (loss + attention_loss_, logits,predict,attention_weights,true_label,prob_label)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        class_labels = features["class_labels"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if FLAGS.crf:
            (total_loss, logits,predicts,attention_weights,true_label,prob_label) = create_model(bert_config, is_training, input_ids,
                                                            mask, segment_ids, label_ids, class_labels, num_labels, 
                                                            use_one_hot_embeddings)

        else:
            (total_loss, logits, predicts,attention_weights,true_label,prob_label) = create_model(bert_config, is_training, input_ids,
                                                            mask, segment_ids, label_ids, class_labels, num_labels, 
                                                            use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        initialized_variable_names=None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits,num_labels,mask):
                predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                # cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels, weights=mask)
                acc = tf.metrics.accuracy(label_ids, predictions)#this is acc for NER and contain "X" label
                return {
                    "acc":acc
                }
                #
            eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:

            predicts_ = {"predicts":predicts,"prob_label":prob_label,"true_label":true_label,"attention_weights":attention_weights}
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts_, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def Writer(predict,true_l,filename,true_token,predict_class_label,maxpredict_prob):
    # if token!="[PAD]" and token!="[CLS]" and true_l!="X":
    if true_token!="[PAD]" and true_token!="[CLS]" and true_token!="**###**":
        if predict=="X" and not predict.startswith("##"):
            predict="O"
        line = "{}\t{}\t{}:{}\t{}\t{}\n".format(true_token,filename,predict_class_label,maxpredict_prob,true_l,predict)
        return line
    else:
        return ''

def draw_plot(tokenizer, attention_weights, sentence):
    f,axs = plt.subplots(figsize=(18,28),nrows=6)
    start=0
    end=0

    for i,axe in enumerate(axs):
        end+=50# [0,50],[50,100],[100,150]
        sent = sentence[start:end]
        sns.heatmap([attention_weights[start:end]],ax=axe)
        axe.set_xticklabels(sent,fontsize=6,rotation=60)
        start=end
    plt.savefig("attention_map.png", format='png',dpi=1000)

def show_confMat(cm,classes_names,title = "Confusion Matrix"):
    """
    show confusion_matrix
    """
    plt.figure()
    plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Paired)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes_names)) 
    plt.xticks(tick_marks,tick_marks) 
    plt.yticks(tick_marks,tick_marks) 
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(x=j,y=i,s=int(cm[i,j]),va='center',ha='center', color='red', fontsize=10)
    plt.ylabel('Predicted Label') 
    plt.xlabel('True Label')
    plt.savefig("Confusion.png", format='png')


def main(_):
    logging.set_verbosity(logging.INFO)
    processors = {"ner": NerProcessor}
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=os.path.dirname(FLAGS.output_dir),
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
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")

        _,_ ,_,_,_= filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_predict:
        with open(FLAGS.middle_output+'/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
   
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(os.path.dirname(FLAGS.output_dir), "predict.tf_record")
        batch_tokens,batch_labels, single_class_label,batch_filename,batch_true_token= filed_based_convert_examples_to_features(predict_examples, label_list,
                                                 FLAGS.max_seq_length, tokenizer,
                                                 predict_file)

        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)
        all_predictions = []
        all_true_labels=[]
        all_attention_sample=[]
        all_predict_prob = []
        predict_probs = []
        true_class_labels = []
        for prel in result:
            predict_probs.append(prel["prob_label"].tolist())
            true_class_labels.append(prel["true_label"].tolist())
            all_predictions.extend(prel["predicts"].tolist())
            all_predict_prob.extend([prel["prob_label"]]*len(prel["predicts"].tolist()))# for write to result file, so here we need use "*"
            all_true_labels.extend([prel["true_label"]]*len(prel["predicts"].tolist()))
            all_attention_sample.append(prel["attention_weights"].tolist())

        # draw attention plot
        draw_plot(tokenizer,all_attention_sample[0],batch_tokens[1:512])
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test_{}.txt".format(FLAGS.disease))
        # here if the tag is "X" means it belong to its before token, here for convenient evaluate use
        # conlleval.pl we  discarding it directly
        with open(output_predict_file,'w') as wf:
            for i,pre in enumerate(all_predictions):
                true_class_label = all_true_labels[i]
                predict_prob = all_predict_prob[i]
                filename = batch_filename[i]
                predict = id2label[pre]
                true_l = id2label[batch_labels[i]]
                true_token = batch_true_token[i]
                line = Writer(predict,true_l,filename,true_token,true_class_label,predict_prob)
                wf.write(line)

        #map predict prob to label
        def mapprob2label(label2id,input,mode="prob"):
            if mode=="prob":
                input = [1 if i>=0.5 else 0 for i in input]
                l = label2id[str(input)]
            elif mode=="label":
                l = label2id[str(input)]
            return l
        true_labels = []
        predict_labels = []
        id2label = labelmap()
        label2id = {str(value):key for key,value in id2label.items()}
        for i,prob in enumerate(predict_probs):
            true_class_label = true_class_labels[i]
            true_class = mapprob2label(label2id,true_class_label,mode="label")
            true_labels.append(true_class)
            predict_class = mapprob2label(label2id,prob,mode="prob")
            predict_labels.append(predict_class)
        precision = precision_score(true_labels,predict_labels, average='macro')
        recall = recall_score(true_labels,predict_labels,average='macro')
        f = f1_score(true_labels,predict_labels,average='macro')

        cm = confusion_matrix(true_labels,predict_labels)
        acc = accuracy_score(true_labels,predict_labels)
        print("\n%%%%%%%%%%%%%%%%%%classfication%%%%%%%%%%%%%%%%%%\n")
        class_loss = "ACC:{:.3f}\nF-Score:{:.3f}\nRecall-Score:{:.3f}\nPrecision-Score:{:.3f}".format(acc,f,precision,recall)
        print(class_loss)
        print("confusion_matrix:\n{}".format(cm))
        show_confMat(cm,[1,2,3,4],title = "Confusion Matrix")
        # print("\n%%%%%%%%%%%%%%%%%%classfication%%%%%%%%%%%%%%%%%%\n")
if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
