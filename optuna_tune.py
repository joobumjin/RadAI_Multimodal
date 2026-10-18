import argparse
from argparse import Namespace

import optuna
from datetime import datetime

import new_kf

def get_arg_parser():
    parser = argparse.ArgumentParser('Optuna Hyperparam Tuning', add_help=False)
    parser.add_argument("--mods",       choices=['prc', 'pr', 'pc', 'rc', 'p', 'r', 'c'], default='prc')
    parser.add_argument("--n_trials",   type=int, default=50)
    parser.add_argument("--merge_obj",  action="store_true")

    return parser

def run_trial(trial: optuna.Trial, args: Namespace):
    """
    A single hyperparameter setup trial as part of an optuna search

    Parameters
    ----------
    trial : optuna.Trial
        an optuna trial to be passed in by an optuna study
    args : Namespace
        parsed CLI args

    Returns
    ----------
    objective_value
        either a merged float value of best value if args.merge_obj
        or a list of float with best AUC and best C-Index
    """
    mod_flag = {
        "p": "--path_lang",
        "r": "--rad_lang",
        "c": "--clinical"
    }
    mod_args = [mod_flag[mod] for mod in args.mods]

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
            '--modality hat --track_best --return_extras'.split() + \
            mod_args

    parser = new_kf.get_args_parser()
    main_fn = new_kf.main

    args = parser.parse_args(args)
    _, res5, _, extras, best = main_fn(args)

    for extra in extras:
        for k, v in extra.items():
            trial.set_user_attr(k, v)

    if args.merge_obj: return best

    return [r.avg for r in res5]


def optimize(args: Namespace):
    """
    Runs the actual optuna hyperparam optimization study

    Parameters
    ----------
    args : Namespace
        parsed CLI args
    """

    mod_names = {
        'p': "Path", 
        'r': "Rad", 
        'c': "Clin"
    }
    mod = "Trimodal" if len(args.mods) == 3 else "".join([mod_names[m] for m in args.mods])

    # get current time and format it as a string
    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if args.merge_obj:
        study = optuna.create_study(directions=["maximize"], storage="sqlite:///db.sqlite3", study_name = f"Trimodal {time_string}")
    else:
        study = optuna.create_study(directions=["maximize", "maximize"], storage="sqlite:///db.sqlite3", study_name = f"{mod} {time_string}")
        study.set_metric_names(["Test AUC", "Test C-Index"])
    study.optimize(lambda trial: run_trial(trial, args), n_trials=args.n_trials, show_progress_bar=True)

    if not args.merge_obj:
        optuna.visualization.plot_pareto_front(study, target_names=["AUC", "C-Index"])

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    optimize(args)