$models = "biomedclip"
# $models = "conch", "biomedclip"
# $fusions = "weighted_sum"
$target = "survival_days"
$loss = "bce"

conda activate multi

# python .\dense_fusion_train.py --model $models --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug
# python .\dense_fusion_train.py --model $models --path_img --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug
# python .\dense_fusion_train.py --model $models --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug
# python .\dense_fusion_train.py --model $models --sparse --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug
# python .\emb_fusion_train.py --model $models --sparse --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug

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

# foreach ($model in $models) {
#     python .\dense_fusion_train.py --model $model --emb_dim 32 --sparse --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
#     $fusions = "naive_sum", "naive_avg", "weighted_sum"
#     foreach ($fusion in $fusions) {   
#         python .\emb_fusion_train.py --model $model --fusion $fusion --sparse --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
#     }
# }

""
"Standard Dense"
# python .\dense_fusion_train.py --debug --loss_fn "cox_nll" --model "gemma" --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss

$models = "gemma", "qwen"
$enc_dims = 768, 1024
for ($i = 0; $i -lt $models.Count; $i++){
    $model, $enc_dim = $models[$i], $enc_dims[$i] 
    python .\dense_fusion_train.py --loss_fn "cox_nll" --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
    $fusions = "naive_sum", "naive_avg", "weighted_sum"
    foreach ($fusion in $fusions) {   
        python .\emb_fusion_train.py --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --fusion $fusion --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
    }
}

# ""
# "Dense Sparse"
# #sparse
# for ($i = 0; $i -lt $models.Count; $i++){
#     $model, $enc_dim = $models[$i], $enc_dims[$i] 
#     python .\dense_fusion_train.py --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --sparse --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
#     $fusions = "naive_sum", "naive_avg", "weighted_sum"
#     foreach ($fusion in $fusions) {   
#         python .\emb_fusion_train.py --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --fusion $fusion --sparse --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
#     }
# }

# ""
# "Curriculum"
# for ($i = 0; $i -lt $models.Count; $i++){
#     $model, $enc_dim = $models[$i], $enc_dims[$i] 
#     python .\curriculum.py  --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss --epochs 500
# }


# #train on RW
# $models = "gemma", "qwen"
# $enc_dims = 768, 1024
# for ($i = 1; $i -lt $models.Count; $i++){
#     $model, $enc_dim = $models[$i], $enc_dims[$i] 
#     python .\dense_fusion_train.py --model $model --emb_dim 128 --test_path "../${model}_multimodal_bins" --train_split 1.0 --run_name "RWTrain-dense-${model}"--data_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --path_lang --rad_lang --label_col $target --loss_fn $loss
#     $fusions = "naive_sum", "naive_avg", "weighted_sum"
#     foreach ($fusion in $fusions) {   
#         python .\emb_fusion_train.py --model $model --emb_dim 128 --test_path "../${model}_multimodal_bins" --train_split 1.0 --run_name "RWTrain-emb_naive-${model}-${fusion}" --data_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --fusion $fusion --path_lang --rad_lang --label_col $target --loss_fn $loss
#     }
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

