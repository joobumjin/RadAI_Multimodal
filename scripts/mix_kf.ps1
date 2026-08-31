conda activate multi

# python .\new_kf.py --disable_wandb --debug --mix_data --clinical --path_lang --rad_lang

#trimodal
python .\new_kf.py --disable_wandb --mix_data --clinical --path_lang --rad_lang

# # bimodal
# python .\new_kf.py --disable_wandb --mix_data --rad_lang --path_lang
# python .\new_kf.py --disable_wandb --mix_data --path_lang --clinical
# python .\new_kf.py --disable_wandb --mix_data --rad_lang --clinical

# #unimodal
# python .\new_kf.py --disable_wandb --mix_data --modality uni_path
# python .\new_kf.py --disable_wandb --mix_data --modality uni_rad
# python .\new_kf.py --disable_wandb --mix_data --modality uni_clin


# # trimodal
# python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --clinical --path_lang --rad_lang

# # bimodal
# python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --rad_lang --path_lang
# python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --path_lang --clinical
# python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --rad_lang --clinical

# # unimodal
# python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --modality uni_path
# python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --modality uni_rad
# python .\new_kf.py --disable_wandb --mix_data --survival_years 3 --modality uni_clin


# trimodal
# python .\new_kf.py --disable_wandb --clinical --path_lang --rad_lang

# #bimodal
# python .\new_kf.py --disable_wandb --rad_lang --path_lang
# python .\new_kf.py --disable_wandb --path_lang --clinical
# python .\new_kf.py --disable_wandb --rad_lang --clinical

# #unimodal
# python .\new_kf.py --disable_wandb --modality uni_path
# python .\new_kf.py --disable_wandb --modality uni_rad
# python .\new_kf.py --disable_wandb --modality uni_clin
