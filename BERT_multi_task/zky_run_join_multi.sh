#!/usr/bin/env bash

filter_tag="True"
do_train="False"
input_dir="data/"
output_dir="../AGACGraphicalModel/data/"

# disease="BLCA"
while getopts i:f:t:o:d:h OPTION
do
    case $OPTION in
        i) input_dir=$OPTARG;;
        f) filter_tag=$OPTARG;;
        t) do_train=$OPTARG;;
        o) output_dir=$OPTARG;;
        d) disease=$OPTARG;;
    \?|h) echo "Usage: $0 [-i input_dir] [-f filter_tag(True)] [-t do_train(True)] [-o output_dir(filter_dir/join_dir)] [-d disease(BLCA)]"
1>&2
    exit 2;;
    esac
done
for line in $(ls $input_dir)
# do
    # echo $line 
#   "filter_mantle_cell_lymphoma.txt"
# for line in "filter_mantle_cell_lymphoma.txt"
do
    disease=${line%.*}
    echo $disease
    python zky_BERT_Joint_NER_CLS.py\
        --task_name="NER"  \
        --do_lower_case=False \
        --crf=False \
        --filter_tag=$filter_tag \
        --do_train=$do_train  \
        --do_predict=True \
        --data_dir=$input_dir  \
        --vocab_file=cased_L-12_H-768_A-12_bert/vocab.txt \
        --bert_config_file=cased_L-12_H-768_A-12_bert/bert_config.json \
        --init_checkpoint=cased_L-12_H-768_A-12_bert/bert_model.ckpt \
        --max_seq_length=512 \
        --train_batch_size=7   \
        --learning_rate=2e-5   \
        --num_train_epochs=1000   \
        --output_dir=./output/$output_dir \
        --disease=$disease \

    perl conlleval.pl -d '\t' < ./output/$output_dir/label_test_$disease.txt

    python zky_get_pmids_function_pairsx.py output/$output_dir $disease
done
