$models = "conch", "biomedclip"
$modalities = "path_only", "rad_only", "sum", "prod"
$target = "death_indicator_2yr"
$loss = "bce"

conda activate multi

foreach ($model in $models) {
    foreach ($modality in $modalities) {
        "Training Simple Predictor on $model embs with $modality modalities"
        python .\text_train.py --model $model --emb_merge $modality --label_col $target --loss_fn $loss
    }
}