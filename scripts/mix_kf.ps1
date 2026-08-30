$target = "survival_days"

conda activate multi

#trimodal
# python .\new_kf.py --disable_wandb --debug --mix_data --survival_years 2 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang --label_col $target
python .\new_kf.py --disable_wandb --mix_data --survival_years 2 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang --label_col $target

#bimodal
# python .\new_kf.py --disable_wandb --mix_data --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --path_lang --label_col $target
# python .\new_kf.py --disable_wandb --mix_data --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --path_lang --clinical --label_col $target
# python .\new_kf.py --disable_wandb --mix_data --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --clinical --label_col $target

# #unimodal
# python .\new_kf.py --disable_wandb --mix_data --modality uni_path --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target
# python .\new_kf.py --disable_wandb --mix_data --modality uni_rad --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target
# python .\new_kf.py --disable_wandb --mix_data --modality uni_clin --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target


# # # trimodal
# # python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang --label_col $target

# # # bimodal
# # python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --path_lang --label_col $target
# # python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --path_lang --clinical --label_col $target
# # python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --clinical --label_col $target

# # # unimodal
# # python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --modality uni_path --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target
# # python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --modality uni_rad --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target
# # python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --modality uni_clin --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target


#trimodal
# python .\new_kf.py --disable_wandb --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang --label_col $target

# #bimodal
# python .\new_kf.py --disable_wandb --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --path_lang --label_col $target
# python .\new_kf.py --disable_wandb --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --path_lang --clinical --label_col $target
# python .\new_kf.py --disable_wandb --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --rad_lang --clinical --label_col $target

# #unimodal
# python .\new_kf.py --disable_wandb --modality uni_path --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target
# python .\new_kf.py --disable_wandb --modality uni_rad --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target
# python .\new_kf.py --disable_wandb --modality uni_clin --loss_fn "cox_nll" --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model gemma --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --label_col $target
