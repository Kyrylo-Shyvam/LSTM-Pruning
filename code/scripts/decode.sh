#!/bin/sh

vocab="data/vocab.json"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="work_dir"

mkdir -p ${work_dir}
echo save results to ${work_dir}

# decoding
python lstmModel.py\
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/"$1" \
    ${test_src} \
    ${test_tgt} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
