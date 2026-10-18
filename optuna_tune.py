import optuna
from datetime import datetime

import new_kf

def run_trial(trial: optuna.Trial):
    """
    A single hyperparameter setup trial as part of an optuna search
    params:
        - trial: an optuna trial to be passed in by an optuna study
    """
    #optimize on lr, batch size, emb dim, fused dim, early stop, num epochs, survival years, activation
    emb_dim = trial.suggest_categorical("emb_dim", [8, 16, 32, 64])
    fusion_layers = trial.suggest_int("fusion_layers", 1, 3)
    fusion_dims = []
    for i in range(fusion_layers):
        out_features = trial.suggest_categorical(f"fusion_l{i}", [8, 16, 32, 64, 128, 256])
        fusion_dims.append(str(out_features))

    # epochs = trial.suggest_categorical("epochs", [200, 300, 500, 800])
    # early_stop = trial.suggest_int("early_stop", low=-100, high=0)
    # warmup = trial.suggest_categorical("warmup", [25, 50, 100])

    #build args, run main
    args =  f'--emb_dim {emb_dim} --fusion_dim {" ".join(fusion_dims)}'.split() + \
            '--disable_wandb --stratify'.split() + \
            '--layer_norm --const_lr --disable_wandb --stratify'.split() + \
            '--clinical --path_lang --rad_lang'.split() + \
            '--modality hat --track_best'.split()

    parser = new_kf.get_args_parser()
    main_fn = new_kf.main

    args = parser.parse_args(args)
    _, res5, _, _ = main_fn(args)
    return [r.avg for r in res5]


def optimize():
    """
    Runs the actual optuna hyperparam optimization study
    """

    # get current time and format it as a string
    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # study = optuna.create_study(directions=["maximize"], storage="sqlite:///db.sqlite3", study_name = f"Trimodal {time_string}")
    study = optuna.create_study(directions=["maximize", "maximize"], storage="sqlite:///db.sqlite3", study_name = f"Trimodal {time_string}")
    study.set_metric_names(["Test AUC", "Test C-Index"])
    study.optimize(run_trial, n_trials=100, show_progress_bar=True)

    optuna.visualization.plot_pareto_front(study, target_names=["AUC", "C-Index"])

if __name__ == '__main__':
    optimize()