import optuna

import new_kf

def run_trial(trial: optuna.Trial):
    #optimize on lr, batch size, emb dim, fused dim, early stop, num epochs, survival years, activation
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    emb_dim = trial.suggest_categorical("emb_dim", [64, 128, 256])
    fused_dim = trial.suggest_categorical("fused_dim", [0, 64, 128])

    early_stop = trial.suggest_int("early_stop", low=-100, high=0)
    epochs = trial.suggest_categorical("epochs", [200, 500])
    warmup = trial.suggest_categorical("warmup", [25, 50, 100])

    act = trial.suggest_categorical("act", ["--rms_norm", "--layer_norm", "--batch_norm"])

    yrs = trial.suggest_categorical("years", [2, 5])

    #build args, run main
    args = f'--survival_years {yrs} {act} --emb_dim {emb_dim} --epochs {epochs} --warmup_epochs {warmup} --early {early_stop} --batch_size {batch_size} --fusion_dim {fused_dim}'.split() + \
            '--disable_wandb --mix_data --clinical --path_lang --rad_lang --sparse --stratify'.split()

    parser = new_kf.get_args_parser()
    main_fn = new_kf.main

    args = parser.parse_args(args)
    _, res5, _ = main_fn(args)

    return [r.avg for r in res5]


def optimize():
    study = optuna.create_study(directions=["maximize", "maximize"], storage="sqlite:///db.sqlite3")
    study.set_metric_names(["Test AUC", "Test C-Index"])
    study.optimize(run_trial, n_trials=100, show_progress_bar=True)

    optuna.visualization.plot_pareto_front(study, target_names=["AUC", "C-Index"])

if __name__ == '__main__':
    optimize()