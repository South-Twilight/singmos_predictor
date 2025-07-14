#!/bin/bash

devices=(0 1 2)
seeds=(0 1234 1984)
pt_datasets=("singmos_v1" "singmos_v2" "singmos_full")
ft_datasets=("singmos_full")
answer_dir=answer
pt_config="config/pretrain.v1.yaml"
ft_config="config/finetune.v1.yaml"

for i in "${!pt_datasets[@]}"; do
    pt_dataset="${pt_datasets[$i]}"
    pids=()
    for j in "${!seeds[@]}"; do
        device="${devices[$j]}"
        (
            echo "RUN $pt_dataset+rd_$seed+device_$device"
            CUDA_VISIBLE_DEVICES=$device ./mdf.sh \
                --pt_dataset $pt_dataset \
                --rd_seed $seed \
                --pt_config $pt_config \
                --ft_config $ft_config \
                --answer_dir $answer_dir
        ) &
        pids+=($!)
    done
    i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;

    python utils/pyutils/merge_metrics.py \
    --src_dir `pwd`/${answer_dir}/"pt_${pt_dataset}_$(basename "${pt_config}" .yaml)"

    for j in "${!ft_datasets[@]}"; do
        ft_dataset="${ft_datasets[$j]}"
        python utils/pyutils/merge_metrics.py \
        --src_dir `pwd`/${answer_dir}/"ft_${ft_dataset}_$(basename "${ft_config}" .yaml)_BASE_pt_${pt_dataset}_$(basename "${pt_config}" .yaml)"
    done
done

echo "All runs are done."

python utils/pyutils/merge_results.py \
--base_path `pwd`/${answer_dir} \
--output_file `pwd`/${answer_dir}/results.csv