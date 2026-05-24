$model = "biomedclip"
# $models = "conch", "biomedclip"
# $fusions = "weighted_sum"
$target = "survival_days"
$loss = "bce"

conda activate multi


# $model = "gemma"
# $enc_dim = 768
# python .\curriculum.py --debug --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss


$models = "gemma", "qwen"
$enc_dims = 768, 1024
for ($i = 0; $i -lt $models.Count; $i++){
    $model, $enc_dim = $models[$i], $enc_dims[$i] 
    python .\curriculum.py --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss --epochs 500
    # $fusions = "naive_sum", "naive_avg", "weighted_sum"
    # foreach ($fusion in $fusions) {   
    #     python .\emb_fusion_train.py --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --fusion $fusion --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
    # }
}

# #sparse
# for ($i = 0; $i -lt $models.Count; $i++){
#     $model, $enc_dim = $models[$i], $enc_dims[$i] 
#     python .\dense_fusion_train.py --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --sparse --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
#     $fusions = "naive_sum", "naive_avg", "weighted_sum"
#     foreach ($fusion in $fusions) {   
#         python .\emb_fusion_train.py --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --fusion $fusion --sparse --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss
#     }
# }
