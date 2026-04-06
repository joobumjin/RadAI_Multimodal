$models = "conch", "biomedclip"
$fusions = "naive_sum", "naive_avg", "weighted_sum"
$target = "death_indicator_2yr"
$loss = "bce"

conda activate multi

# python .\logit_fusion_train.py --model conch --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --debug

foreach ($model in $models) {
    python .\logit_fusion_train.py --model $model --clinical  --label_col $target --loss_fn $loss
}

foreach ($model in $models) {
    foreach ($fusion in $fusions) {
        python .\logit_fusion_train.py --model $model --clinical --path_lang --fusion $fusion --label_col $target --loss_fn $loss
    }
}


foreach ($model in $models) {
    foreach ($fusion in $fusions) {
        python .\logit_fusion_train.py --model $model --clinical --rad_lang --fusion $fusion --label_col $target --loss_fn $loss
    }
}


foreach ($model in $models) {
    foreach ($fusion in $fusions) {
        python .\logit_fusion_train.py --model $model --clinical --path_lang --rad_lang --fusion $fusion --label_col $target --loss_fn $loss
    }
}