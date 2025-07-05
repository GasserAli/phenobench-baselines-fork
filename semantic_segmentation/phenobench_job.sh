#block(name=pheno_job, threads=3, memory=25000, subtasks=1, gpus=1, hours=8)
	echo $CUDA_VISIBLE_DEVICES
	source /home/malek.ahmed/miniconda3/etc/profile.d/conda.sh
	conda activate phenobench_semseg
	python3 train.py --config ./config/config_deeplab.yaml --export_dir ./exports
