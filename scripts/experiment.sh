# FIRST OF ALL, YOU NEED TO DOWNLOAD THE FEATURES OF THE THREE DATASET FROM:
# https://github.com/yabufarha/ms-tcn

# create gt arrays
python utils/generate_gt_array.py --dataset_dir [DATASET_DIR]
python utils/generate_boundary_array.py --dataset_dir [DATASET_DIR]

# make csv files for training and testing
python utils/make_csv_files.py --dataset_dir [DATASET_DIR]

# make configuration files
python utils/make_config.py --root_dir ./config/DeST_linearformer/MCFS-130 --dataset MCFS-130 --split 1

# MCFS-130 dataset
# training
python train.py ./config/DeST_linearformer/MCFS-130/config.yaml

# test
python evaluate.py ./config/DeST_linearformer/MCFS-130/config.yaml

# average cross validation results.
python utils/average_cv_results.py ./result/MCFS-130/DeST_linearformer

# output visualization
# save outputs as np.array
python save_pred.py ./config/DeST_linearformer/MCFS-130/config.yaml
# convert array to image
python utils/convert_arr2img.py ./result/MCFS-130/DeST_linearformer/split1/predictions
