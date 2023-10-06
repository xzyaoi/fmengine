singularity run --nv \
--home /mnt/ds3lab-scratch/xiayao:/home/xiayao \
--bind /mnt/ds3lab-scratch/xiayao/cache/HF/hub:/.hf_cache \
--env HF_HOME=/.hf_cache \
--bind .cache/pretrained_weights:/pretrained \
--bind $PWD:/workspace \
--pwd /workspace \
fmsys.sif \
python scripts/conversions/llama/from_hf.py --model_name_or_path openlm-research/open_llama_3b_v2 --output_dir .cache/pretrained_weights/open_llama_3b_mp4 --mp_world_size 4