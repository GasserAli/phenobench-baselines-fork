#block(name=pheno_test, threads=3, memory=25000, subtasks=1, gpus=1, hours=8)
	echo $CUDA_VISIBLE_DEVICES
	source ~/miniconda3/etc/profile.d/conda.sh
	conda activate phenobench_semseg
    python3 ./semantic_segmentation/test.py --config ./semantic_segmentation/config/config_deeplab.yaml  --ckpt_path ./semantic-seg-deeplab.ckpt --export_dir ./exports/test