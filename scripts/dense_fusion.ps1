conda activate multi

""
"Standard Dense"
# python .\dense_fusion_train.py --debug --clinical --path_lang --rad_lang


python .\dense_fusion_train.py --disable_wandb --run_name "Full Multi" --mix_data --clinical --path_lang --rad_lang 
# python .\dense_fusion_train.py --run_name "Bimodal" --mix_data --clinical --path_lang
# python .\dense_fusion_train.py --clinical --path_lang --rad_lang
# $fusions = "naive_sum", "naive_avg", "weighted_sum"
# foreach ($fusion in $fusions) {   
#     python .\emb_fusion_train.py --fusion $fusion --clinical --path_lang --rad_lang
# }

# ""
# "Curriculum"
#     python .\curriculum.py --clinical --path_lang --rad_lang --epochs 500

# #sparse 
# # clinical only
#     python .\dense_fusion_train.py --sparse --clinical

# # langauge only
#     python .\dense_fusion_train.py --sparse --path_lang --rad_lang

# # clin + path lang
#     python .\dense_fusion_train.py --sparse --clinical --path_lang

# # clin + rad lang
#     python .\dense_fusion_train.py --sparse --clinical --rad_lang

# #all 3
#     python .\dense_fusion_train.py --sparse --clinical --path_lang --rad_lang