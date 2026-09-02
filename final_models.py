import torch
from torch import nn

from util import *
from model import create_mlp, DenseFusion, Unimodal

def get_clinical_encoder(args):
    # clin_enc = create_mlp(24, [128], args.emb_dim, act = nn.GELU(), dropout = 0.3, layer_norm = True)
    clin_enc = create_mlp(24, [], args.emb_dim, act = nn.GELU(), dropout = 0.3, end_with_fc=False, end_with_dropout=True, rms_norm = True, end_with_norm=True)
    return clin_enc, False

def get_path_lang_encoder(args):
    # path_lang_enc = create_mlp(args.enc_dim, [128], args.emb_dim, act = nn.GELU(), dropout = 0.4, layer_norm = True)
    path_lang_enc = create_mlp(args.enc_dim, [], args.emb_dim, act = nn.GELU(), dropout = 0.3, end_with_fc=False, end_with_dropout=True, rms_norm = True, end_with_norm=True)
    return path_lang_enc, True

def get_rad_lang_encoder(args):
    # rad_lang_enc = create_mlp(args.enc_dim, [128], args.emb_dim, act = nn.GELU(), dropout = 0.4, layer_norm = True)
    rad_lang_enc = create_mlp(args.enc_dim, [], args.emb_dim, act = nn.GELU(), dropout = 0.3, end_with_fc=False, end_with_dropout=True, rms_norm = True, end_with_norm=True)
    return rad_lang_enc, True

def get_dense_fusion_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    get_enc_fns = {
        "clinical": get_clinical_encoder, 
        "clinical_imputed": get_clinical_encoder, 
        "path_lang": get_path_lang_encoder, 
        "rad_lang": get_rad_lang_encoder, 
    }

    encs, casts = {}, {}
    arg_dict = vars(args)
    for mod, fn in get_enc_fns.items():
        if arg_dict.get(mod, False): 
            encs[mod], casts[mod] = fn(args)

    model = DenseFusion(encs, emb_dim=args.emb_dim, hidden_dims=[32], autocast=casts, loss_fn=LOSS_FNS[args.loss_fn], device=device)
    return model, device

# --------------------------------------------------------

def get_unimodal_path(args):
    model = Unimodal(args.enc_dim, [128, 32], 1, act = nn.GELU(), loss_fn=LOSS_FNS[args.loss_fn], rms_norm = True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return model, device

def get_unimodal_rad(args):
    model = Unimodal(args.enc_dim, [128, 32], 1, act = nn.GELU(), loss_fn=LOSS_FNS[args.loss_fn], rms_norm = True, mod="rad_lang")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return model, device

def get_unimodal_clin(args):
    model = Unimodal(24, [64], 1, act = nn.GELU(), loss_fn=LOSS_FNS[args.loss_fn], rms_norm = True, mod="clinical") 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    return model, device