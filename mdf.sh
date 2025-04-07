#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=2
# dataset related
data_dir=data/SingMOS_
pt_dataset=
ft_dataset=singmos_full
# train related
pt_config=config/pretrain.v1.yaml
ft_config=config/finetune.v1.yaml
rd_seed=1234
output_dir=ckpt
# infer related
checkpoint=valid.best.pth
answer_dir=answer

. utils/parse_options.sh

pt_dir="pt_${pt_dataset}_$(basename "${pt_config}" .yaml)_${rd_seed}"
ft_dir="ft_${ft_dataset}_$(basename "${pt_config}" .yaml)_${rd_seed}_BASE_${pt_dir}"

echo $pt_dir
echo $ft_dir

mkdir -p ${output_dir}
mkdir -p ${answer_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: pretrain stage on ${pt_dataset}"

    outdir=${output_dir}/${pt_dir}
    ansdir=${answer_dir}/${pt_dir}

    python train.py \
        --datadir ${data_dir} \
        --dataname ${pt_dataset} \
        --model_config ${pt_config} \
        --outdir ${outdir} \
        --seed ${rd_seed}

    python infer.py \
        --datadir ${data_dir} \
        --model_config ${pt_config} \
        --ckpt ${outdir}/${checkpoint} \
        --answer_dir ${ansdir} \
        --seed ${rd_seed}

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: finetune stage on ${ft_dataset}"

    outdir=${output_dir}/${ft_dir}
    ansdir=${answer_dir}/${ft_dir}

    python train.py \
        --datadir ${data_dir} \
        --dataname ${ft_dataset} \
        --model_config ${ft_config} \
        --outdir ${outdir} \
        --seed ${rd_seed}

    python infer.py \
        --datadir ${data_dir} \
        --model_config ${pt_config} \
        --ckpt ${outdir}/${checkpoint} \
        --answer_dir ${ansdir} \
        --seed ${rd_seed}

fi