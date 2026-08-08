$target = "survival_days"

conda activate multi


""
"Standard Dense"
# python .\dense_fusion_train.py --debug --loss_fn "cox_nll" --model "gemma" --data_path "../gemma_multimodal_bins" --test_path "../gemma_multimodal_bins_rw" --enc_dim 768 --clinical --path_lang --rad_lang --label_col $target --loss_fn $loss

# $models = "gemma", "qwen"
$models = "gemma"
$enc_dims = 768, 1024
for ($i = 0; $i -lt $models.Count; $i++){
    $model, $enc_dim = $models[$i], $enc_dims[$i] 
    python .\dense_fusion_train.py --modality uni_path --run_name uni_path_gemma --loss_fn "cox_nll" --mix_data --batch_size 128 --lr 1e-3 --warmup_epochs 25 --model $model --data_path "../${model}_multimodal_bins" --test_path "../${model}_multimodal_bins_rw" --enc_dim $enc_dim --clinical --path_lang --rad_lang --label_col $target
}
