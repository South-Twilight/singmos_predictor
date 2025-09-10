# EXP1

### SSL
CUDA_VISIBLE_DEVICES=0 python train.py \
    --datadir "DATA" \
    --dataname "SingEval" \
    --model_config config/SSL.yaml \
    --outdir checkpoints/SSL

### SSL + Domain ID
CUDA_VISIBLE_DEVICES=1 python train.py \
    --datadir "DATA" \
    --dataname "SingEval" \
    --model_config config/SSL+domain_id.yaml \
    --outdir checkpoints/SSL+DOMAIN_ID


### SSL + MDF
CUDA_VISIBLE_DEVICES=2 python train.py \
    --datadir "DATA" \
    --dataname "SingEval" \
    --model_config config/SSL+single_dataset.yaml \
    --outdir checkpoints/SSL+MDF/pt_single_dataset


CUDA_VISIBLE_DEVICES=2 python train.py \
    --datadir "DATA" \
    --dataname "SingEval" \
    --model_config config/SSL.yaml \
    --outdir checkpoints/SSL+MDF/ft_full_dataset \
    --finetune_from_ckpt checkpoints/SSL+MDF/pt_single_dataset/latest.pth


### SSL + MDF + Domain ID
CUDA_VISIBLE_DEVICES=3 python train.py \
    --datadir "DATA" \
    --dataname "SingEval" \
    --model_config config/SSL+single_dataset+domain_id.yaml \
    --outdir checkpoints/SSL+MDF+DOMAIN_ID/pt_single_dataset


CUDA_VISIBLE_DEVICES=3 python train.py \
    --datadir "DATA" \
    --dataname "SingEval" \
    --model_config config/SSL+domain_id.yaml \
    --outdir checkpoints/SSL+MDF+DOMAIN_ID/ft_full_dataset
    --finetune_from_ckpt checkpoints/SSL+MDF+DOMAIN_ID/pt_single_dataset/latest.pth

### Infer

CUDA_VISIBLE_DEVICES=4 python infer.py \
    --datadir DATA \
    --model_config config/SSL.yaml \
    --ckpt checkpoints/SSL/latest.pth \
    --answer_dir answer/SSL \
    --test_sets 'singeval_p1' \
    --test_sets 'singeval_p2' \
    --test_sets 'singeval_p3' 
