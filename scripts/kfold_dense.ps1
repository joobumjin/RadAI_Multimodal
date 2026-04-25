$models = "biomedclip"
# $models = "conch", "biomedclip"
# $fusions = "weighted_sum"
$target = "death_indicator_2yr"
$loss = "bce"

conda activate multi

# python .\kfold_dense_fusion_train.py --model $models --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug

# # clinical only
# foreach ($model in $models) {
#     python .\kfold_dense_fusion_train.py --model $model --clinical  --label_col $target --loss_fn $loss
# }

# # langauge only
# foreach ($model in $models) {
#     python .\kfold_dense_fusion_train.py --model $model --path_lang --rad_lang --label_col $target --loss_fn $loss
# }

# # clin + path lang
# foreach ($model in $models) {
#     python .\kfold_dense_fusion_train.py --model $model --clinical --path_lang --label_col $target --loss_fn $loss
# }

# # clin + rad lang
# foreach ($model in $models) {
#     python .\kfold_dense_fusion_train.py --model $model --clinical --rad_lang --label_col $target --loss_fn $loss
# }

#all 3
foreach ($model in $models) {
    python .\kfold_dense_fusion_train.py --model $model --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss --folds 6 --epochs 400
}
