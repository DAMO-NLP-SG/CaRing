export DATASET_NAME="proofwriter"

export INPUT_PATH="../data/${DATASET_NAME}/meta-test.shuffled.json"
export DEMO_PATH="llm_prompts_related/icl_demonstrations/${DATASET_NAME}/"

export MAX_SAMPLES=-1

export MODE="$1"

export DEMO_INDICES="1 2"
# export DEMO_INDICES="1 2 3" # for CoT only

export LLM="CodeLlama-34b-hf"
export NUM_GPUS=2



OUTPUT_PATH="../output/${DATASET_NAME}/${MODE}.${LLM}.num_samples-${MAX_SAMPLES}.json"
# export MODEL_PATH="../../../pretrained_models/${LLM}"

python run_generation.py \
    --input_path ${INPUT_PATH} \
    --output_path ${OUTPUT_PATH} \
    --demonstration_path ${DEMO_PATH} \
    --model_name_or_path ../../../../pretrained_models/${LLM} \
    --demo_indices ${DEMO_INDICES} \
    --max_tokens 2048 \
    --num_gpus ${NUM_GPUS} \
    --mode ${MODE} \
    --max_samples ${MAX_SAMPLES} \
    --overwrite_output

    # --max_tokens_total 4096 \