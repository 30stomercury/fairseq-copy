set -ex
export FAIRSEQ_ROOT=/lustre/s2196654/fairseq
export KALDI_ROOT=/lustre/s2196654/pykaldi/tools/kaldi


<< EOF
RVAD_ROOT=.
subset=dev-clean
for subset in train-clean-100 dev-clean test-clean dev-other test-other
do
    manifest_path=../manifest/wav2vec/$subset
    original_audio_path=/home/s2196654/dataset/LibriSpeech/$subset
    audio_path=exp/$subset/wav
    mkdir -p $audio_path
    mkdir -p $manifest_path
    echo $manifest_path
    # create a manifest file for the set original of audio files
    python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py $original_audio_path --ext flac --dest $manifest_path --valid-percent 0
    python scripts/vads.py -r $RVAD_ROOT < $manifest_path/train.tsv > $audio_path/$subset.vads
    python scripts/remove_silence.py --tsv $manifest_path/train.tsv --vads $audio_path/$subset.vads --out $audio_path
    python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py $audio_path --ext flac --dest manifest/wav2vec/$subset-vad --valid-percent 0.0
done
EOF


libri_dir=/home/s2196654/dataset/LibriSpeech/
matched_path=exp/train-clean-100/wav2vec_vox_new/matched
unmatched_path=exp/train-clean-100/wav2vec_vox_new/unmatched
subset=train-clean-100
mkdir -p $matched_path
mkdir -p $unmatched_path
cp manifest/wav2vec/train-clean-100-vad/train.tsv  $matched_path/train.tsv
cp manifest/wav2vec/dev-clean-vad/train.tsv  $matched_path/valid.tsv
cp manifest/wav2vec/test-clean-vad/train.tsv  $matched_path/test.tsv

# Audio
# 512 pca dim, layer 14
#zsh scripts/prepare_audio.sh $matched_path $matched_path/feat hub/wav2vec2.0-large/libri960_big.pt 512 14
#zsh scripts/prepare_audio.sh $matched_path $matched_path/feat hub/wav2vec2.0-large/wav2vec_vox_new.pt 512 14
#zsh scripts/prepare_audio.sh $matched_path $matched_path/feat hub/hubert-base/hubert_base_ls960.pt 512 9

# Text
#wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
#mv lid.176.bin exp/train-clean-100/wav2vec2_large_960/
## unmached 
#for text in train-clean-360 train-other-500
#do
#    find $libri_dir/$text -type f -name "*.txt" -exec cat {} + | cut -d ' ' -f 2- \
#        >> $unmatched_path/860hr-texts.txt
#done
#zsh scripts/prepare_text.sh en $unmatched_path/860hr-texts.txt $unmatched_path 10 G2P exp/train-clean-100/wav2vec2_large_960/lid.176.bin 0.25

## matched texts: otherwise no validation error shown
#split=valid
#for text in dev-clean
#do
#    cat $matched_path/$split.tsv | \
#        awk '{print $1}' | \
#        sed -E "s@([0-9]+)/([0-9]+)-[0-9]+-([0-9]+)\.flac@grep \3 ${libri_dir}/${text}/\2/\1/\2-\1.trans.txt@" | tail -n +2 > \
#        $matched_path/grep_texts.sh
#    sh $matched_path/grep_texts.sh | cut -d ' ' -f 2- > $matched_path/$split.wrd
#    rm $matched_path/grep_texts.sh
#done
#sh scripts/prepare_text.sh en $matched_path/$split.wrd $matched_path 0 G2P exp/train-clean-100/wav2vec2_large_960/lid.176.bin 0.0
#python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 0 --lexicon $matched_path/lexicon.lst < $matched_path/$split.wrd > $matched_path/$split.phn
#wc -l $matched_path/lexicon.lst
#wc -l $matched_path/$split.wrd
#wc -l $matched_path/$split.phn
#cp $matched_path/$split.phn $matched_path/feat/precompute_pca512_cls128_mean_pooled/
#cp ${PWD}/${matched_path}/phones/dic* ${PWD}/${matched_path}/feat/precompute_pca512_cls128_mean_pooled
#rm -r $matched_path/phones
#rm $matched_path/dict.txt

#split=test
#for text in test-clean
#do
#    cat $matched_path/$split.tsv | \
#        awk '{print $1}' | \
#        sed -E "s@([0-9]+)/([0-9]+)-[0-9]+-([0-9]+)\.flac@grep \3 ${libri_dir}/${text}/\2/\1/\2-\1.trans.txt@" > \
#        $matched_path/grep_texts.sh
#    sh $matched_path/grep_texts.sh | cut -d ' ' -f 2- > $matched_path/$split.wrd
#    rm $matched_path/grep_texts.sh
#    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 0 --surround --lexicon $unmatched_path/lexicon_filtered.lst < $matched_path/$split.wrd > $matched_path/$split.phn
#done

# Experiments
PREFIX=w2v_unsup_gan_xp

# For wav2vec-U, audio features are pre-segmented
#CONFIG_NAME=w2vu
#TASK_DATA=${PWD}/${matched_path}/feat/precompute_pca512_cls128_mean_pooled
CONFIG_NAME=w2vu-hsmm
#TASK_DATA=/lustre/s2196654/results/20ms/unsupervised-asr/linear-hsmm/train-clean-100/hubert-l9/3/decoded_feats/pooled/train-clean-100/
#TASK_DATA=/lustre/s2196654/results/20ms/unsupervised-asr/linear-hsmm/train-clean-100/hubert-l9/3/decoded_feats/pooled_large/train-clean-100/
#TASK_DATA=/lustre/s2196654/results/20ms/unsupervised-asr/linear-hsmm/train-clean-100/hubert-l9-gold/0/decoded_feats/pooled_large/train-clean-100/
TASK_DATA=/home/s2196654/results/20ms/unsupervised-asr/linear-hsmm/train-clean-100/hubert-l9-unitrans/2/decoded_feats/pooled_large/train-clean-100/

# Unpaired text input
TEXT_DATA=${PWD}/${unmatched_path}/phones  # path to fairseq-preprocessed GAN data (phones dir)
KENLM_PATH=${PWD}/exp/train-clean-100/wav2vec2_large_960/unmatched/phones/lm.phones.filtered.04.bin #${PWD}/${unmatched_path}/phones/lm.phones.filtered.04.bin

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    -m --config-dir config/gan \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    optimizer.groups.generator.optimizer.lr='[0.00005]' \
    optimizer.groups.generator.lr=\[0.00005\] \
    optimizer.groups.discriminator.optimizer.lr='[0.0003]' \
    optimizer.groups.discriminator.lr=\[0.0003\] \
    model.generator_batch_norm_init_stats=${TASK_DATA} \
    model.discriminator_kernel=8 \
    model.generator_kernel=9 \
    model.code_penalty=3.0 model.gradient_penalty=1.0 \
    model.smoothness_weight=1.5 'common.seed=range(0,5)'


#for seed in 0
#do
#python w2vu_generate.py --config-dir config/generate --config-name viterbi \
#fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
#fairseq.task.data=${TASK_DATA} \
#fairseq.common_eval.path=${PWD}/multirun/2024-10-16/01-20-20/$seed/checkpoint_last.pt \
#fairseq.dataset.gen_subset=valid results_path=multirun/2024-10-16/01-20-20/$seed/results
#done
#
