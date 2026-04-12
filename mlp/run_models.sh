# pip install -r requirements.txt

echo "--- Input: MRI, Target: DEMENTED ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality mri \
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

echo "\n--- Input: AMYLOID_PET, Target: DEMENTED ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality amyloid_pet \
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

echo "\n--- Input: TAU_PET, Target: DEMENTED ---"
python runner.py \
--data_dir ../data \
--save_model \
--save_dir outputs \
--modality tau_pet \
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
