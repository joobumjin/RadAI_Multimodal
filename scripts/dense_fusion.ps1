$models = "biomedclip"
# $models = "conch", "biomedclip"
# $fusions = "weighted_sum"
$target = "survival_days"
$loss = "bce"

conda activate multi

# python .\dense_fusion_train.py --model $models --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug
# python .\dense_fusion_train.py --model $models --path_img --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug
# python .\dense_fusion_train.py --model $models --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug
python .\dense_fusion_train.py --model $models --sparse --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug

# # clinical only
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --clinical  --label_col $target --loss_fn $loss
# }

# # langauge only
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --path_lang --rad_lang --label_col $target --loss_fn $loss
# }

# # clin + path lang
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --clinical --path_lang --label_col $target --loss_fn $loss
# }

# # clin + rad lang
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --clinical --rad_lang --label_col $target --loss_fn $loss
# }

#all 3
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
# }


# #sparse 
# # clinical only
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --sparse --clinical  --label_col $target --loss_fn $loss
# }

# # langauge only
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --sparse --path_lang --rad_lang --label_col $target --loss_fn $loss
# }

# # clin + path lang
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --sparse --clinical --path_lang --label_col $target --loss_fn $loss
# }

# # clin + rad lang
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --sparse --clinical --rad_lang --label_col $target --loss_fn $loss
# }

# #all 3
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --sparse --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
# }


# # imputed clinical
# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --clinical_imputed  --label_col $target --loss_fn $loss
# }

# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --clinical_imputed --path_lang --label_col $target --loss_fn $loss
# }

# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --clinical_imputed --rad_lang --label_col $target --loss_fn $loss
# }

# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --clinical_imputed --path_lang --rad_lang --label_col $target --loss_fn $loss
# }

