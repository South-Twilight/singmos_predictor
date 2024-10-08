#!/bin/bash

# model_configs=("v0.yaml" "v00.yaml" "v000.yaml" "v0000.yaml" "v00000.yaml" "v000000.yaml" "v0000000.yaml")
# model_configs=("v1.yaml" "v11.yaml" "v111.yaml")
model_configs=("hb.v0.yaml" "hl.v0.yaml" "w2vl.v0.yaml" "wavlmb.v0.yaml" "wavlml.v0.yaml" "xlsr.v0.yaml")
# seeds=(1984 2301 3906 4918)
seeds=(4918)
devices=(0 1 2 4 5 6)


# 遍历所有模型配置文件和种子组合
for i in "${!model_configs[@]}"; do
    config="${model_configs[$i]}"
    vid=$(echo "$config" | sed 's/\.v0.yaml$//')
    device="${devices[$i]}"
    echo "run $config with CUDA:$device"
    for seed in "${seeds[@]}"; do
        echo "Running: python infer.py --model_config $config --seed $seed"
        CUDA_VISIBLE_DEVICES="$device" python infer.py \
            --datadir data/voicemos2024/DATA \
            --model_config "config/iscslp/s3prl/${config}" \
            --finetuned_checkpoint "ckpt/iscslp2/ckpt_${vid}" \
            --answer_dir "answer/iscslp2" \
            --seed "$seed" &
    done
done

wait

echo "All runs are done."