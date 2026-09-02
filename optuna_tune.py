import optuna

import new_kf

def run_trial(trial: optuna.Trial):
    #optimize on lr, batch size, emb dim, fused dim, early stop, num epochs, survival years, activation
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    emb_dim = trial.suggest_categorical("emb_dim", low=64, high=256, step=64)
    fused_dim = trial.suggest_int("fused_dim", low=0, high=128, step=64)

    early_stop = trial.suggest_int("early_stop", low=-100, high=0)
    epochs = trial.suggest_categorical("epochs", [200, 500])
    warmup = trial.suggest_categorical("warmup", [25, 50, 100])

    act = trial.suggest_categorical("act", ["--rms_norm", "--layer_norm", "--batch_norm"])

    #build args, run main
    yrs = trial.suggest_categorical("years", [2, 5])
    args = f'--survival_years {yrs} {act} --emb_dim {emb_dim} --epochs {epochs} --warmup_epochs {warmup} --early_stop {early_stop} --batch_size {batch_size} --fusion_dim {fused_dim}'.split() + \
            '--disable_wandb--mix_data --clinical --path_lang --rad_lang --sparse --stratify'.split()

    parser = new_kf.get_args_parser()
    main_fn = new_kf.main

    args = parser.parse_args(args)
    _, res5, _ = main_fn(args)

    return [r.avg for r in res5]


def optimize():
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(run_trial, n_trials=100)

    optuna.visualization.plot_pareto_front(study, target_names=["AUC", "C-Index"])

if __name__ == '__main__':
    optimize()