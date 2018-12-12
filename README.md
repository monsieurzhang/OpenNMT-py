# General

Here records the code and training methods for ...


Table of Contents
=================
  * [Data](#Data)
  * [Training](#Training)
  * [Generating backtranslated data](#Generating backtranslated data)
  * [Run on FloydHub](#run-on-floydhub)
  * [Citation](#citation)

## Data

We use the data from [tensor2tensor](https://github.com/tensorflow/tensor2tensor/blob/71371c7f40f1110159b81b46b4bbca7006996c22/tensor2tensor/data_generators/translate_ende.py#L62) for En-De translations [directly download](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8).

|         | Training | Dev (newstest2013) | Test (newstest2014) | Test (newstest2015) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| #sents  | 4500966  | 3000 | 3003 | 2169 |

## Training

- Data binarization:

```bash
python preprocess.py -train_src $DATA_FOLDER/train.tok.clean.bpe.32000.en -train_tgt $DATA_FOLDER/train.tok.clean.bpe.32000.de -valid_src $DATA_FOLDER/newstest2013.tok.bpe.32000.en -valid_tgt $DATA_FOLDER/newstest2013.tok.bpe.32000.de -save_data $DATA_FOLDER_BIN/bin
```

- Basic configs:

```bash
python  train.py -data $DATA_FOLDER_BIN/bin -save_model $MODEL_FOLDER/model \
        -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
        -encoder_type transformer -decoder_type transformer -position_encoding \
        -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
        -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
        -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
        -max_grad_norm 0 -param_init 0  -param_init_glorot \
        -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
        -world_size 2 -gpu_ranks 0 1
```

We train both En-De and De-En for 200K iterations as baseline systems. The De-En system is then used for back translation.
We use the same config for training. The only difference is the data we used for each system (baseline, backtrans, etc.)

## Generating backtranslated data

Basically, we translate all the training sentences (De) to generate synthetic En sentences.
The common decoding is like:

```bash
python translate.py \
  -gpu 0 \
  -model $MODEL_FOLDER/model_step_200000.pt \
  -src $DATA_FOLDER/train.tok.clean.bpe.32000.de \
  -output $WORKINGDIR/[].bt.en \
  -replace_unk
```
We regard this as the "$COMMON_DECODING". The generated file name will follow the convention "[].bt.en".

### Follow the work from [Understanding Back-Translation at Scale](https://arxiv.org/pdf/1808.09381)

- beam

```bash
$COMMON_DECODING
```

- greedy

```bash
$COMMON_DECODING \
  -beam_size 1
```

- top10

```bash
$COMMON_DECODING \
  -sampling_topk 10
```

- sampling

```bash
$COMMON_DECODING \
  -sampling
```

- beam+noise

```bash
cat beam.bt.en | python $ONMT/tools/addnoise.py > beam_noise.bt.en
```
