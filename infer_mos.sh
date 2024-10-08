src_path="/data3/tyx/task/yf_mos/data1"
tgt_path="infer_mos/v3"

if [ ! -e "${tgt_path}/DATA" ]; then
    mkdir -p "${tgt_path}/DATA"
    mkdir -p "${tgt_path}/DATA/sets"
fi

> "$tgt_path/answer.txt"

for data_dir in ${src_path}/* ; do
    # data process
    data_dir_name=$(basename "${data_dir}")
    for sys_dir in ${data_dir}/* ; do
        sys_dir_name=$(basename "${sys_dir}")
        sys_data=$(echo "${data_dir_name}@${sys_dir_name}")
        echo $sys_dir
        ln -snf ${sys_dir} "$tgt_path/DATA/wav"
        > "$tgt_path/DATA/sets/eval_mos_list.txt"
        wav_list=${sys_dir}/*
        for wav in ${wav_list}; do
            wav_path=$(basename "${wav}")
            echo "${wav_path},1" >> "$tgt_path/DATA/sets/eval_mos_list.txt" 
        done

        # predict MOS
        CUDA_VISIBLE_DEVICES=7 python singmos/infer.py \
        --model_config config/config_wav2vec2_small.yaml \
        --datadir $tgt_path/DATA \
        --finetuned_checkpoint checkpoints-v0/ckpt_15 \
        --outfile $tgt_path/answer.txt \
        --sys ${sys_data} \
        --appdix $tgt_path/details
    done
done
