# Run prepare_training_data for dunlemn head detection
- make sure your source path folder look like this structure:
  src_path/task_id/images/{Dataset_Name}/*.jpg
                  /annotations/{YOUR_Datumaro_FORMAT.json}
- step1:pip install loguru
- step2:export PYTHONPATH=$PYTHONPATH:$PWD && python3 prepare_training_data.py --src_path /data/training/source/head/extract/1 --dst_path /data/training/prepare/head --target_names head