$models = "conch", "biomedclip"
# $fusions = "naive_sum", "naive_avg", "weighted_sum"
$fusions = "weighted_sum"
$target = "death_indicator_2yr"
$loss = "bce"

conda activate multi

# python .\logit_fusion_train.py --model conch --path_lang --clinical --rad_lang --label_col $target --loss_fn $loss --fusion weighted_sum --debug

# clinical only
foreach ($model in $models) {
    python .\logit_fusion_train.py --model $model --clinical  --label_col $target --loss_fn $loss
}

# langauge only
foreach ($model in $models) {
    foreach ($fusion in $fusions) {
        python .\logit_fusion_train.py --model $model --path_lang --rad_lang --fusion $fusion --label_col $target --loss_fn $loss
    }
}

# clin + path lang
foreach ($model in $models) {
    foreach ($fusion in $fusions) {
        python .\logit_fusion_train.py --model $model --clinical --path_lang --fusion $fusion --label_col $target --loss_fn $loss
    }
}

# clin + rad lang
foreach ($model in $models) {
    foreach ($fusion in $fusions) {
        python .\logit_fusion_train.py --model $model --clinical --rad_lang --fusion $fusion --label_col $target --loss_fn $loss
    }
}

#all 3
foreach ($model in $models) {
    foreach ($fusion in $fusions) {
        python .\logit_fusion_train.py --model $model --clinical --path_lang --rad_lang --fusion $fusion --label_col $target --loss_fn $loss
    }
}

