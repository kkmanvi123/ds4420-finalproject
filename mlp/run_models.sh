#pip install -r requirements.txt

echo "Testing pipeline..."
echo "\n--- Input: MRI, Target: DEMENTED ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality mri \
--target_col DEMENTED \
--task classification \
--n_components 30 \
--epochs 20 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 256,128,64 \
--dropout 0.3 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-4

echo "\n--- Input: FDG_PET, Target: DEMENTED ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality fdg_pet \
--target_col DEMENTED \
--task classification \
--n_components 0.95 \
--epochs 20 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 256,128,64 \
--dropout 0.3 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-4

# Amyloid and Tau are not the strongest candidates for AD predictions
# echo "\n--- Input: AMYLOID_PET, Target: DEMENTED ---"
# python runner.py \
# --data_dir ../data \
# --save_model \
# --save_dir outputs \
# --modality amyloid_pet \
# --target_col DEMENTED \
# --task classification \
# --n_components 0.95 \
# --epochs 20 \
# --batch_size 64 \
# --lr 1e-3 \
# --hidden_dims 256,128,64 \
# --dropout 0.3 \
# --activation relu \
# --use_batchnorm \
# --weight_decay 1e-4

# echo "\n--- Input: TAU_PET, Target: DEMENTED ---"
# python runner.py \
# --data_dir ../data \
# --save_model \
# --save_dir outputs \
# --modality tau_pet \
# --target_col DEMENTED \
# --task classification \
# --n_components 0.95 \
# --epochs 20 \
# --batch_size 64 \
# --lr 1e-3 \
# --hidden_dims 256,128,64 \
# --dropout 0.3 \
# --activation relu \
# --use_batchnorm \
# --weight_decay 1e-4

# Classificaion candidates
echo "\nGetting classification candidates"
echo "\n--- Input: FDG_PET, Target: MEMORY ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality fdg_pet \
--target_col MEMORY \
--task classification \
--n_components 0.95 \
--epochs 20 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 256,128,64 \
--dropout 0.3 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-4 \
--run_name "fdg_pet_MEMORY_classification_baseline"

echo "\n--- Input: FDG_PET, Target: CDRLANG ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality fdg_pet \
--target_col CDRLANG \
--task classification \
--n_components 0.95 \
--epochs 20 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 256,128,64 \
--dropout 0.3 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-4

# echo "\n--- Input: MRI, Target: MOCA ---"
# python runner.py \
# --data_dir ../data \
# --save_model \
# --save_dir outputs \
# --modality mri \
# --target_col NACCMOCA \
# --task classification \
# --n_components 30 \
# --epochs 20 \
# --batch_size 64 \
# --lr 1e-3 \
# --hidden_dims 256,128,64 \
# --dropout 0.3 \
# --activation relu \
# --use_batchnorm \
# --weight_decay 1e-4

# Regression candidates
echo "\nGetting regression candidates"
echo "\n--- Input: MRI, Target: CDRSUM ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality mri \
--target_col CDRSUM \
--task regression \
--n_components 30 \
--epochs 20 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 256,128,64 \
--dropout 0.3 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-4

echo "\n--- Input: MRI, Target: MOCA ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality mri \
--target_col NACCMOCA \
--task regression \
--n_components 0.95 \
--epochs 20 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 256,128,64 \
--dropout 0.3 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-4 \
--run_name "mri_NACCMOCA_regression_baseline" 

# Start the tuning
echo "\nTuning the models..."

# Tuning: FDG_PET + MEMORY classification
# Baseline: fdg_pet_MEMORY_classification_baseline

echo "\n--- Tuning: FDG_PET, Target: MEMORY | tune1 ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality fdg_pet \
--target_col MEMORY \
--task classification \
--n_components 0.90 \
--epochs 30 \
--batch_size 64 \
--lr 5e-4 \
--hidden_dims 256,128,64 \
--dropout 0.4 \
--activation relu \
--use_batchnorm \
--weight_decay 5e-4 \

echo "\n--- Tuning: FDG_PET, Target: MEMORY | tune2 ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality fdg_pet \
--target_col MEMORY \
--task classification \
--n_components 0.95 \
--epochs 30 \
--batch_size 32 \
--lr 3e-4 \
--hidden_dims 512,256,128 \
--dropout 0.3 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-4 \

echo "\n--- Tuning: FDG_PET, Target: MEMORY | tune3 ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality fdg_pet \
--target_col MEMORY \
--task classification \
--n_components 0.98 \
--epochs 25 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 128,64 \
--dropout 0.5 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-3 \

echo "\n--- Input: FDG_PET, Target: MEMORY | tune4 ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality fdg_pet \
--target_col MEMORY \
--task classification \
--n_components 0.95 \
--epochs 25 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 256,128,64 \
--dropout 0.3 \
--activation leaky_relu \
--use_batchnorm \
--weight_decay 1e-4 \
--run_name "fdg_pet_MEMORY_classification_baseline_leakyrelu"

# Tuning: MRI + NACCMOCA regression
# Baseline: mri_NACCMOCA_regression_baseline

echo "\n--- Tuning: MRI, Target: NACCMOCA | tune1 ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality mri \
--target_col NACCMOCA \
--task regression \
--n_components 30 \
--epochs 30 \
--batch_size 64 \
--lr 5e-4 \
--hidden_dims 256,128,64 \
--dropout 0.4 \
--activation relu \
--use_batchnorm \
--weight_decay 5e-4

echo "\n--- Tuning: MRI, Target: NACCMOCA | tune2 ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality mri \
--target_col NACCMOCA \
--task regression \
--n_components 50 \
--epochs 35 \
--batch_size 32 \
--lr 3e-4 \
--hidden_dims 512,256,128 \
--dropout 0.3 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-4

echo "\n--- Tuning: MRI, Target: NACCMOCA | tune3 ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality mri \
--target_col NACCMOCA \
--task regression \
--n_components 0.95 \
--epochs 25 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 128,64 \
--dropout 0.5 \
--activation relu \
--use_batchnorm \
--weight_decay 1e-3

echo "\n--- Input: MRI, Target: MOCA | tune4---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality mri \
--target_col NACCMOCA \
--task regression \
--n_components 0.95 \
--epochs 25 \
--batch_size 64 \
--lr 1e-3 \
--hidden_dims 256,128,64 \
--dropout 0.3 \
--activation gelu \
--use_batchnorm \
--weight_decay 1e-4 \
--run_name "mri_NACCMOCA_regression_baseline_gelu" 