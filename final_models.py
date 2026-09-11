import torch
from torch import nn

from util import *
from model import create_mlp, DenseFusion, Unimodal

def get_clinical_encoder(args):
    act = {
        "batch_norm": args.batch_norm,
        "rms_norm": args.rms_norm,
        "layer_norm": args.layer_norm,
    }

    clin_enc = create_mlp(24, [], args.emb_dim, act = nn.GELU(), dropout = 0.2, end_with_fc=False, end_with_dropout=True, end_with_norm=True, **act)
    return clin_enc

def get_path_lang_encoder(args):
    act = {
        "batch_norm": args.batch_norm,
        "rms_norm": args.rms_norm,
        "layer_norm": args.layer_norm,
    }

    path_lang_enc = create_mlp(args.enc_dim, [], args.emb_dim, act = nn.GELU(), dropout = 0.2, end_with_fc=False, end_with_dropout=True, end_with_norm=True, **act)
    return path_lang_enc

def get_rad_lang_encoder(args):
    act = {
        "batch_norm": args.batch_norm,
        "rms_norm": args.rms_norm,
        "layer_norm": args.layer_norm,
    }

    rad_lang_enc = create_mlp(args.enc_dim, [], args.emb_dim, act = nn.GELU(), dropout = 0.2, end_with_fc=False, end_with_dropout=True, end_with_norm=True, **act)
    return rad_lang_enc

def get_dense_fusion_model(args, dtypes):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    get_enc_fns = {
        "clinical": get_clinical_encoder, 
        "clinical_imputed": get_clinical_encoder, 
        "path_lang": get_path_lang_encoder, 
        "rad_lang": get_rad_lang_encoder, 
    }

    act = {
        "batch_norm": args.batch_norm,
        "rms_norm": args.rms_norm,
        "layer_norm": args.layer_norm,
    }

    encs, casts = {}, {}
    arg_dict = vars(args)
    for mod, fn in get_enc_fns.items():
        if arg_dict.get(mod, False): 
            encs[mod] = fn(args)
            casts[mod] = (dtypes[f"{mod}_dtype"] != "float32")

    hid = [args.fusion_dim, 32] if args.fusion_dim > 0 else [32]

    model = DenseFusion(encs, emb_dim=args.emb_dim, hidden_dims=hid, autocast=casts, loss_fn=LOSS_FNS[args.loss_fn], device=device, **act)
    return model, device

# --------------------------------------------------------

def get_unimodal_path(args, dtypes):
    model = Unimodal(args.enc_dim, [128, 32], 1, act = nn.GELU(), loss_fn=LOSS_FNS[args.loss_fn], rms_norm = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return model, device

def get_unimodal_rad(args, dtypes):
    model = Unimodal(args.enc_dim, [128, 32], 1, act = nn.GELU(), loss_fn=LOSS_FNS[args.loss_fn], rms_norm = True, mod="rad_lang")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return model, device

def get_unimodal_clin(args, dtypes):
    model = Unimodal(24, [64, 32], 1, act = nn.GELU(), loss_fn=LOSS_FNS[args.loss_fn], rms_norm = True, mod="clinical") 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return model, device

# --------------------------------------------------------

MODEL_FNS = {
    "multi": get_dense_fusion_model,
    "uni_path": get_unimodal_path,
    "uni_rad": get_unimodal_rad,
    "uni_clin": get_unimodal_clin
}

# --------------------------------------------------------

def get_mod_name(args):
    if args.modality == "uni_path": 
        args.path_lang = True
        mod_name = "Unimodal Path"
    elif args.modality == "uni_rad": 
        args.rad_lang = True
        mod_name = "Unimodal Rad"
    elif args.modality == "uni_clin": 
        args.clinical = True
        mod_name = "Unimodal Clin"
    else:
        mod_name = []
        for mod, n in zip([args.path_lang, args.rad_lang, args.clinical], ["Path", "Rad", "Clin"]):
            if mod: mod_name.append(n)
        
        if len(mod_name) == 3: mod_name = "Trimodal"
        else: mod_name = "Bimodal " + "+".join(mod_name)

    return args, mod_name