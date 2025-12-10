CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file config/step4.yaml train_judge_sft.py
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file config/step4.yaml train_20epoch.py