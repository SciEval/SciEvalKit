#!/bin/bash

# # Set proxy environment variables
# for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do 
#     export $v=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128; 
# done

# Load necessary modules
module load anaconda/2021.11
module load cuda/11.7.0
module load cudnn/8.6.0.163_cuda11.x
module load compilers/gcc/9.3.0
module load llvm/triton-clang_llvm-11.0.1

# Activate the Conda environment
source activate dnalm_v2

# Set environment variables
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/peng_tmp_test/miniconda3/lib
export CPATH=/usr/include:$CPATH
export PYTHONUNBUFFERED=1
export LD_PRELOAD=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64/libstdc++.so.6

# SLURM configuration
nproc_per_node=1
partition='ai4bio'   #ailab
quotatype='vip_gpu_ailab'  #vip_gpu_ailab_low
run_type='srun'  # choice between [srun, sbatch]

# Model and Task Variables
# model=alpaca-7b
# model=BioMedGPT-LM-7B
# model=galactica
# model=glm-4-9b-chat
# model=gpt-4o
# model=gpt-4o-mini
# model=InstructProtein
# model=Llama-2-7b-chat-hf
# model=llama-molinstruct-protein-7b
# model=Meta-Llama-3-8B-Instruct
# model=llama3.1
# model=qwen2-7b
# model=vicuna-7b-v1.5
# model=our_model_without_pretrain_1
model=chat_multi_omics
# for model in Llama-2-7b-chat-hf Meta-Llama-3-8B-Instruct gpt-4o-mini qwen2-7b llama-molinstruct-protein-7b glm-4-9b-chat galactica
# do
OMICS=all_omics
task=evaluation_metrics
job_name=${model}_${task}
# input_file_path=./model_output_for_eval/${model}_response_${OMICS}.jsonl
input_file_path=./model_output_for_eval/chat_multi_omics_text.json
master_port=$(shuf -i 10000-45000 -n 1)

# Choose sbatch or srun
if [ "$run_type" == "srun" ]; then
    EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${model}_${task}  --gres=gpu:$nproc_per_node --cpus-per-task=4 torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"           
else
    EXEC_PREFIX="${run_type} --nodes=1 -p ${quotatype} -A ${partition} --job-name=${model}_${task}  --gres=gpu:$nproc_per_node --cpus-per-task=4  --output=logging/${model}_$(date +"%Y%m%d_%H%M%S") torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"          
fi    

echo "Executing on GPU device: $gpu_device" 
echo "Model: $model, Task: $task"
echo "Executing with: ${EXEC_PREFIX}"

# Run the Python file
${EXEC_PREFIX} \
evaluate.py --model_name ${model} --OMICS ${OMICS} --input_file_path ${input_file_path}
# done