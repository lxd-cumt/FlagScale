# sparse model change the parallel config
python convert.py \
    --model-type mixtral \
    --loader transformers \
    --saver mcore \
    --load-dir Mixtral-8x7B-v0.1 \
    --save-dir output \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 2 \
    --target-params-dtype fp32 \


# dense model convert to sparse model, mlp weight copy to all experts weight
# padding vocab_size with default value 64
python convert.py \
    --model-type mistral mixtral \
    --loader transformers \
    --saver mcore \
    --load-dir Mistral-7B-v0.1 \
    --save-dir output \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 2 \
    --target-params-dtype fp32 \
    --target-num-experts 8 \
    --true-vocab-size 151851 \


python convert.py \
    --model-type aquila3_dense aquila3_moe \
    --loader transformers \
    --saver mcore \
    --load-dir /share/project/aquila3/FlagScale/outputs/checkpoints/ \
    --save-dir output-moe-mcore \
    --true-vocab-size 151851 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 2 \
    --target-num-experts 8 \
#    --target-params-dtype fp32 \


python convert.py \
    --model-type aquila3_moe \
    --loader mcore \
    --saver transformers \
    --load-dir output-moe-mcore \
    --save-dir output-moe-hf \
    --true-vocab-size 151851 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
#    --target-params-dtype bf16


# for emu-7b: aquila3-dense-hf --> emu3-dense-mg
python tools/checkpoint/convert.py \
    --model-type aquila3_dense emu3_dense \
    --loader transformers \
    --saver mcore \
    --load-dir /share/project/zhaoyingli/checkpoints/aquila3_7b_nocooling_hf \
    --save-dir /share/project/zhaoyingli/checkpoints/ckpt_emu3_7b/ \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --true-vocab-size 184622 \
    --true-language-vocab-size 151851 \
    --target-params-dtype fp32 \
    --build-model-with-initialization \


# for emu-7b: emu3-dense-mg --> emu3-dense-mg
python tools/checkpoint/convert.py \
    --model-type emu3_dense \
    --loader mcore \
    --saver mcore \
    --load-dir /share/project/zhaoyingli/checkpoints/ckpt_emu3_7b \
    --save-dir /share/project/zhaoyingli/checkpoints/ckpt_emu3_7b_2 \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --true-vocab-size 184622 \
    --true-language-vocab-size 151851 \
    --target-params-dtype fp32 \


# for emu-7b: emu3-dense-mg --> emu3-dense-hf
python tools/checkpoint/convert.py \
    --model-type emu3_dense \
    --loader mcore \
    --saver transformers \
    --load-dir /share/project/zhaoyingli/checkpoints/ckpt_emu3_7b_2/ \
    --save-dir /share/project/zhaoyingli/checkpoints/hf_ckpt_emu3_7b \
    --target-params-dtype bf16 \
    --true-vocab-size 128256 \
    --megatron-path <xxx>

python convert.py \
    --model-type llama \
    --loader transformers \
    --saver mcore \
    --load-dir ${transformers_ckpt_path:??} \
    --save-dir ${mcore_ckpt_path:??} \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 4 \
    --target-expert-parallel-size 1 \
    --max-queue-size 50 \
    --target-params-dtype bf16 \
    --true-vocab-size 128256 \
    --megatron-path <xxx>

python convert.py \
    --model-type aquila3_dense \
    --loader transformers \
    --saver mcore \
    --load-dir $loaddir \
    --save-dir $outputs \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 151665 \
    --megatron-path <xxx>

# megatron to huggingface
python convert.py \
    --model-type llama \
    --loader mcore \
    --saver transformers \
    --load-dir $loaddir \
    --save-dir $outputs \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 1 \
    --target-params-dtype bf16 \
    --true-vocab-size 128256 \
    --megatron-path <xxx>
    --true-vocab-size 184622 \
    --true-language-vocab-size 151851 \


# # for emu-8x7b
# python tools/checkpoint/convert.py \
#     --model-type aquila3_moe emu3_moe \
#     --loader mcore \
#     --saver mcore \
#     --load-dir /share/project/zhaoyingli/gitee/aquila_7b_k73_qwen_aquila3_gama_moe/ \
#     --save-dir /share/project/zhaoyingli/gitee/ckpt_emu3_8x7b/ \
#     --target-tensor-parallel-size 1 \
#     --target-pipeline-parallel-size 8 \
#     --target-expert-parallel-size 2 \
#     --true-vocab-size 168237 \
#     --true-language-vocab-size 151851 \
#     --target-num-experts 8 \
#     --build-model-with-initialization \


# # for multimodal emu-12x7b
# python tools/checkpoint/convert.py \
#     --model-type aquila3_moe emu3_moe \
#     --loader mcore \
#     --saver mcore \
#     --load-dir /share/project/zhaoyingli/gitee/aquila_7b_k73_qwen_aquila3_gama_moe/ \
#     --save-dir /share/project/zhaoyingli/gitee/ckpt_emu3_12x7b/ \
#     --target-tensor-parallel-size 1 \
#     --target-pipeline-parallel-size 8 \
#     --target-expert-parallel-size 2 \
#     --true-vocab-size 168237 \
#     --true-language-vocab-size 151851 \
#     --target-num-experts 12 \
#     --target-num-experts-split 8 \
#     --use-multimodal-router \
#     --build-model-with-initialization \
