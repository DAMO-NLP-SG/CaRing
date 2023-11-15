
export ROOT_DIR=..

export DATASET_NAME=gsm8k

export DATASET_PATH="${ROOT_DIR}/data/${DATASET_NAME}/"
export DEMO_PATH="llm_prompts_related/icl_demonstrations/${DATASET_NAME}/"
export DEMO_INDICES="1 2 3 4 5"

export INPUT_FILE="reasoning_annotated_test.jsonl"


export MAX_SAMPLES=-1

export NUM_GPUS=2

export MODE="$1"

export LLM="CodeLlama-34b-hf"

export OUTPUT_PATH=${ROOT_DIR}/output/${DATASET_NAME}/${MODE}.${LLM}.num_samples-${MAX_SAMPLES}.json

export MODEL_PATH=../../../../pretrained_models/${LLM}

python run_generation.py \
    --input_path ${DATASET_PATH}${INPUT_FILE} \
    --output_path ${OUTPUT_PATH} \
    --demonstration_path ${DEMO_PATH} \
    --model_name_or_path ${MODEL_PATH} \
    --demo_indices ${DEMO_INDICES} \
    --max_tokens 2048 \
    --num_gpus ${NUM_GPUS} \
    --mode ${MODE} \
    --max_samples ${MAX_SAMPLES} \
    --overwrite_output

    # --max_tokens_total 4096 \