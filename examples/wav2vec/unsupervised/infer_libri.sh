#set -ex
export FAIRSEQ_ROOT=/home/s2196654/fairseq-copy
matched_path=exp/train-clean-100/wav2vec_vox_new/matched
unmatched_path=exp/train-clean-100/wav2vec_vox_new/unmatched

UNIT=phone
if [[ "$UNIT" == "phone" ]]; then
    lm_weight=2.0
    word_score=-3
    beam_size=10
    python3 scripts/generate_lexicon.py -i $unmatched_path/phones/dict.phn.txt -o $unmatched_path/phones/lexicon.phone.lst
    TARGET_DATA=${PWD}/exp/train-clean-100/wav2vec2_large_960/matched/valid.phn
    LEXICON_PATH=${PWD}/$unmatched_path/phones/lexicon.phone.lst
    KENLM_PATH=${PWD}/exp/train-clean-100/wav2vec2_large_960/unmatched/phones/lm.phones.filtered.04.bin #${PWD}/${unmatched_path}/phones/lm.phones.filtered.04.bin
    #TASK_DATA=${PWD}/${matched_path}/feat/precompute_pca512_cls128_mean #_pooled
    TASK_DATA=/home/s2196654/results/20ms/unsupervised-asr/linear-hsmm/train-clean-100/hubert-l9-unitrans/2/decoded_feats/wav2vec2_pooled_large/train-clean-100/
    config_name=kaldi-phn
elif [[ "$UNIT" == "word" ]]; then
    lm_weight=2.0
    word_score=-3
    beam_size=10
    TARGET_DATA=${PWD}/exp/train-clean-100/wav2vec2_large_960/matched/valid.wrd
    LEXICON_PATH=${PWD}/$unmatched_path/phones/lexicon_filtered.lst
    KENLM_PATH=${PWD}/exp/train-clean-100/wav2vec2_large_960/unmatched/phones/kenlm.wrd.o40003.bin
    TASK_DATA=${PWD}/${matched_path}/feat/precompute_pca512_cls128_mean
    config_name=kaldi-wrd
else
    echo "Error: Invalid unit type '$UNIT'"
    exit 1
fi


# w2v-u
#ckpt_path=multirun/2024-10-23/17-25-48/
# hsmm
#ckpt_path=multirun/2024-11-13/13-03-33/
ckpt_path=multirun/2025-01-30/23-13-34/
cp -n ${PWD}/${matched_path}/feat/precompute_pca512_cls128_mean_pooled/valid.phn ${TASK_DATA}
cp -n ${PWD}/${unmatched_path}/phones/dict.txt ${TASK_DATA}/dict.phn.txt


# === Print Configuration ===
echo "matched_path: $matched_path"
echo "unmatched_path: $unmatched_path"
echo "KENLM_PATH: $KENLM_PATH"
echo "LEXICON_PATH: $LEXICON_PATH"
echo "TASK_DATA: $TASK_DATA"
echo "TARGET_DATA: $TARGET_DATA"
echo "ckpt: $ckpt_path"
echo "dict: ${PWD}/${unmatched_path}/phones/dict.txt"


for seed in 0
do
python3 w2vu_generate.py --config-dir config/generate --config-name $config_name \
    fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    fairseq.task.data=${TASK_DATA} \
    fairseq.common_eval.path=${PWD}/$ckpt_path/$seed/checkpoint_best.pt \
    fairseq.dataset.gen_subset=valid results_path=${PWD}/$ckpt_path/$seed/results \
    targets=${TARGET_DATA} \
    post_process=silence \
    lm_model=${KENLM_PATH} \
    kenlm_model=${KENLM_PATH} \
    lm_weight=$lm_weight \
    lexicon=${LEXICON_PATH} \
    word_score=$word_score \
    beam=$beam_size
done
