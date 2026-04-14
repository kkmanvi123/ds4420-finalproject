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
--n_components 30 \
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
--weight_decay 1e-4