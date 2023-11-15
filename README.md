# Neuro-Symbolic Integration Brings Causal and Reliable Reasoning Proofs

This repo contains the code for the paper [Neuro-Symbolic Integration Brings Causal and Reliable Reasoning Proofs](https://github.com/ringos/RingoS.github.io/tree/master/files/better_reasoning_proof-preprint.pdf)


## Table of Contents

- [Overview](https://github.com/DAMO-NLP-SG/CaRing#overview)
- [Environment Setup](https://github.com/DAMO-NLP-SG/CaRing#environment-setup)
- [Usage](https://github.com/DAMO-NLP-SG/CaRing#usage)
- [Citation](https://github.com/DAMO-NLP-SG/CaRing#citation)
- [Acknowledgements](https://github.com/DAMO-NLP-SG/CaRing#acknowledgments)


## Overview

CaRing (*Ca*usal and *R*eliable Reason*ing*) is a neuro-symbolic integration method for compelx reasoning problems.
The main advantage of CaRing is that it produces strictly causal and reliable reasoning proofs along with the answer, allowing humans to understand the reasoning process of the model and verify the correctness and safety of the answer.

## Environment Setup

### 1. Set Up Conda Environment (Required for prompting LLMs)


```shell
conda create -n caring python=3.9
conda activate caring

git clone https://github.com/DAMO-NLP-SG/CaRing.git
cd CaRing

# We use cuda=11.8, you may change it to your cuda version
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

Here are some problems that you may encounter when installing the environment: [\[Issue-1\]](https://github.com/vllm-project/vllm/issues/731#issuecomment-1675387132).

### 2. Install SWI-Prolog (Required for evaluation)

**Note**: Below installation steps follow the guide from the [SWI-Prolog official page](https://www.swi-prolog.org/download/stable).

* **For Ubuntu Machines**:
    ```shell
    sudo apt-get update
    sudo apt-get install software-properties-common

    sudo apt-add-repository ppa:swi-prolog/stable
    sudo apt-get update
    sudo apt-get install swi-prolog

    pip install git+https://github.com/yuce/pyswip@master#egg=pyswip
    ```

* **Other Linux Machines**:

    See the [SWI-Prolog official page](https://www.swi-prolog.org/download/stable).



## Usage
### Prompt LLMs
```shell
cd CaRing
mkdir output

export MODE=prolog  # Our Method
# export MODE=cot  # CoT

sh scripts/run_proofwriter.sh ${MODE}
# sh scripts/run_gsm8k.sh ${MODE}
# sh scripts/run_prontoqa.sh ${MODE}
```

You may customize the configurations in the corresponding `.sh` scripts.
Currently we support Code-LLaMA series as the base LLM.

### Evaluation:
* ProofWriter
    ```shell
    export MODE=prolog  # Our Method
    # export MODE=cot  # CoT
    python evaluate.py \
        --gpt_response_path ../output/proofwriter/${MODE}.CodeLlama-34b-hf.num_samples--1.demo-1_2.json \
        --dataset_name proofwriter \
        --evaluate_proof \
        --use_multiprocessing
    ```
* GSM8K
    ```shell
    export MODE=prolog  # Our Method
    # export MODE=cot  # CoT
    python evaluate.py \
        --gpt_response_path ../output/gsm8k/${MODE}.CodeLlama-34b-hf.num_samples--1.demo-1_2_3_4_5.json \
        --dataset_name gsm8k \
        --evaluate_proof \
        --use_multiprocessing
    ```
* PrOntoQA
    ```shell
    export MODE=prolog  # Our Method
    # export MODE=cot  # CoT
    python evaluate.py \
        --gpt_response_path ../output/prontoqa/${MODE}.CodeLlama-34b-hf.num_samples--1.demo-1_2.json \
        --dataset_name prontoqa \
        --use_multiprocessing
    ```

## Citation

If you find this project useful, please consider citing our work:
```
@article{yang2023neuro,
  title={Neuro-Symbolic Integration Brings Causal and Reliable Reasoning Proofs},
  author="Yang, Sen  and
      Li, Xin and
      Cui, Leyang and
      Bing, Lidong and
      Lam, Wai",
  journal={arXiv preprint},
  year={2023}
}
```

## Acknowledgments
This project makes use of the following open-source projects:
* [Code-LLaMA](https://github.com/facebookresearch/codellama)
* [vLLM](https://github.com/vllm-project/vllm)
* [SWI-Prolog](https://www.swi-prolog.org/)
* [PySwip](https://github.com/yuce/pyswip)