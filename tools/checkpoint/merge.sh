
# NOTE!!!!!!
# target-expert-parallel-size must be 1 for dense model
python convert.py \
    --model-type aquila3_dense \
    --loader transformers \
    --saver mcore \
    --load-dir /share/project/zhaoyingli/checkpoints/aquila3_7b_nocooling_hf/ \
    --save-dir /share/project/zhaoyingli/checkpoints/ckpt_aquila3_7b_tp2pp2/ \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 1 \
    --true-vocab-size 151851 \
    --target-params-dtype bf16 \


# NOTE!!!!!!
# target-expert-parallel-size must be 1 for dense model
python convert.py \
    --model-type emu3_dense \
    --loader mcore \
    --saver mcore \
    --load-dir /share/project/zhaoyingli/checkpoints/v4_emu3_7b_stage1_bs5120_lr0001_180k/ \
    --save-dir /share/project/zhaoyingli/checkpoints/ckpt_emu3_7b_tp2pp2/ \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 1 \
    --true-vocab-size 184622 \
    --true-language-vocab-size 151851 \
    --target-params-dtype bf16 \


# NOTE!!!!!!
# the src-model-types's models must have the exactly same configurations
# target-num-experts-split is the number of experts in aquila3_dense to expand
# use-attn-type is the attention type to use in the merged model, 
#     if use "aquila3_dense", the merged model will has qkv_bias from aquila3_dense
#     if use "emu3_dense", the merged model won't have qkv_bias.
#     if use "mean", the merge model won't have qkv_bias, and qkv_linear and proj_linear will be the mean of the two models
python merge.py \
    --src-model-types aquila3_dense emu3_dense \
    --dst-model-type emu3_moe \
    --load-dirs "/share/project/zhaoyingli/checkpoints/ckpt_aquila3_7b_tp2pp2/" "/share/project/zhaoyingli/checkpoints/ckpt_emu3_7b_tp2pp2/" \
    --save-dir /share/project/zhaoyingli/checkpoints/ckpt_emu3_8x7b_tp2pp2_merged/ \
    --target-num-experts 8 \
    --target-num-experts-split 4 \
    --use-attn-type "emu3_dense" # aquila3_dense or emu3_dense or mean


python convert.py \
    --model-type emu3_moe \
    --loader mcore \
    --saver mcore \
    --load-dir /share/project/zhaoyingli/checkpoints/ckpt_emu3_8x7b_tp2pp2_merged/ \
    --save-dir /share/project/zhaoyingli/checkpoints/ckpt_emu3_tp2pp2ep2_final/ \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 2 \
    --true-vocab-size 184622 \
    --true-language-vocab-size 151851 \
    --target-params-dtype bf16 \
