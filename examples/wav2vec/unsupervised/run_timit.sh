export FAIRSEQ_ROOT=/lustre/s2196654/fairseq
ssl=wav2vec2_large_ll60k
<< EOF
mkdir -p hub/wav2vec2.0-large
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt
mv libri960_big.pt hub/wav2vec2.0-large/
mkdir -p exp/timit/$ssl


# timit
bash scripts/prepare_timit.sh ../../../../dataset/timit/timit-wav exp/timit/$ssl  hub/wav2vec2.0-large/libri960_big.pt 
EOF

PREFIX=w2v_unsup_gan_xp
cp ${PWD}/exp/timit/$ssl/matched/feat/dic* ${PWD}/exp/timit/$ssl/matched/feat/precompute_pca512_cls128_mean_pooled

# For wav2vec-U, audio features are pre-segmented
CONFIG_NAME=w2vu
TASK_DATA=${PWD}/exp/timit/$ssl/matched/feat/precompute_pca512_cls128_mean_pooled

# Unpaired text input
TEXT_DATA=${PWD}/exp/timit/$ssl/unmatched/phones  # path to fairseq-preprocessed GAN data (phones dir)
KENLM_PATH=/lustre/s2196654/espnet/egs2/timit/uasr1/exp/ngram/4gram.bin

#PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
#    -m --config-dir config/gan \
#    --config-name $CONFIG_NAME \
#    task.data=${TASK_DATA} \
#    task.text_data=${TEXT_DATA} \
#    task.kenlm_path=${KENLM_PATH} \
#    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
#    model.code_penalty=2 model.gradient_penalty=1.5 \
#    model.smoothness_weight=0.5 'common.seed=range(0,5)'

for seed in 0 1 2 3 4
do
python w2vu_generate.py --config-dir config/generate --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=${TASK_DATA} \
fairseq.common_eval.path=${PWD}/multirun/2024-10-12/20-51-55/$seed/checkpoint_last.pt \
fairseq.dataset.gen_subset=valid results_path=multirun/2024-10-12/20-51-55/results
done
