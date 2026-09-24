from argparse import Namespace

import torch
from torch import nn

from util import *
from model import create_mlp, DenseFusion, Unimodal, AttentionFusion, HierarchicalAttentionFusion

def norms(args: Namespace) -> dict:
    return {
        "batch_norm": args.batch_norm,
        "rms_norm": args.rms_norm,
        "layer_norm": args.layer_norm,
    }

# --------------------------------------------------------

def get_clinical_encoder(args: Namespace) -> nn.Module:
    "Get the Clinical Modality Encoder from args"

    setup = {
        "end_with_fc": False, 
        "end_with_dropout": True, 
        "end_with_norm": True,
        **norms(args)
    }

    clin_enc = create_mlp(in_dim = 24, hid_dims = [], out_dim = args.emb_dim, act = nn.GELU(), dropout = 0.2, **setup)
    return clin_enc

def get_path_lang_encoder(args) -> nn.Module:
    "Get the PathLang Modality Encoder from args"

    setup = {
        "end_with_fc": False, 
        "end_with_dropout": True, 
        "end_with_norm": True,
        **norms(args)
    }

    path_lang_enc = create_mlp(in_dim = args.enc_dim, hid_dims = [], out_dim = args.emb_dim, act = nn.GELU(), dropout = 0.2, **setup)
    return path_lang_enc

def get_rad_lang_encoder(args) -> nn.Module:
    "Get the RadLang Modality Encoder from args"

    setup = {
        "end_with_fc": False, 
        "end_with_dropout": True, 
        "end_with_norm": True,
        **norms(args)
    }

    rad_lang_enc = create_mlp(in_dim = args.enc_dim, hid_dims = [], out_dim = args.emb_dim, act = nn.GELU(), dropout = 0.2, **setup)
    return rad_lang_enc

enc_fns = {
    "clinical": get_clinical_encoder, 
    "clinical_imputed": get_clinical_encoder, 
    "path_lang": get_path_lang_encoder, 
    "rad_lang": get_rad_lang_encoder, 
}

# --------------------------------------------------------

def get_dense_fusion_model(args: Namespace, dtypes:dict[str,str]) -> tuple[nn.Module, torch.device]:
    "Get the Dense+Cat intermediate fusion model with the single modality encoders"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encs, casts = {}, {}
    arg_dict = vars(args)
    for mod, fn in enc_fns.items():
        if arg_dict.get(mod, False): 
            encs[mod] = fn(args)
            casts[mod] = (dtypes[mod] != "float32")

    model = DenseFusion(encs, emb_dim=args.emb_dim, hidden_dims=args.fusion_dim, 
                        autocast=casts, loss_fn=LOSS_FNS[args.loss_fn], device=device, 
                        **norms(args))
    model = model.to(device)
    return model, device

def get_attn_fusion_model(args: Namespace, dtypes:dict[str,str]) -> tuple[nn.Module, torch.device]:
    "Get the Keyless Attention fusion model with the single modality encoders"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encs, casts = {}, {}
    arg_dict = vars(args)
    for mod, fn in enc_fns.items():
        if arg_dict.get(mod, False): 
            encs[mod] = fn(args)
            casts[mod] = (dtypes[mod] != "float32")

    model = AttentionFusion(encs, emb_dim=args.emb_dim, hidden_dims=args.fusion_dim, 
                            autocast=casts, loss_fn=LOSS_FNS[args.loss_fn], device=device, 
                            **norms(args))
    model = model.to(device)
    return model, device

def get_ha_fusion_model(args: Namespace, dtypes:dict[str,str]) -> tuple[nn.Module, torch.device]:
    "Get the HierarchicalKeyless Attention fusion model with the single modality encoders"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encs, casts = {}, {}
    arg_dict = vars(args)
    for mod, fn in enc_fns.items():
        if arg_dict.get(mod, False): 
            encs[mod] = fn(args)
            casts[mod] = (dtypes[mod] != "float32")

    model = HierarchicalAttentionFusion(
        encs, emb_dim=args.emb_dim, hidden_dims=args.fusion_dim, 
        autocast=casts, loss_fn=LOSS_FNS[args.loss_fn], device=device, 
        **norms(args)
    )
    model = model.to(device)
    return model, device

# --------------------------------------------------------

def get_unimodal_path(args: Namespace, dtypes:dict[str,str]) -> tuple[nn.Module, torch.device]:
    "Get the Unimodal PathLang predictor model from args"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cast = dtypes["path_lang"]!="float32"
    
    model = Unimodal(args.enc_dim, args.fusion_dim, 1, act = nn.GELU(), 
                     loss_fn=LOSS_FNS[args.loss_fn], autocast=cast, **norms(args))
    model = model.to(device)
    return model, device

def get_unimodal_rad(args: Namespace, dtypes:dict[str,str]) -> tuple[nn.Module, torch.device]:
    "Get the Unimodal RadLang predictor model from args"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cast = dtypes["rad_lang"]!="float32"

    model = Unimodal(args.enc_dim, args.fusion_dim, 1, act = nn.GELU(), 
                     loss_fn=LOSS_FNS[args.loss_fn], mod="rad_lang", autocast=cast, 
                     **norms(args))
    model = model.to(device)
    return model, device

def get_unimodal_clin(args: Namespace, dtypes:dict[str,str]) -> tuple[nn.Module, torch.device]:
    "Get the Unimodal Clinical predictor model from args"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cast = dtypes["clinical"]!="float32"

    model = Unimodal(24, args.fusion_dim, 1, act = nn.GELU(), 
                     loss_fn=LOSS_FNS[args.loss_fn], mod="clinical", autocast=cast, 
                     **norms(args))
    model = model.to(device)
    return model, device

# --------------------------------------------------------

MODEL_FNS = {
    "multi": get_dense_fusion_model,
    "attn": get_attn_fusion_model,
    "hat": get_ha_fusion_model,
    "uni_path": get_unimodal_path,
    "uni_rad": get_unimodal_rad,
    "uni_clin": get_unimodal_clin
}

# --------------------------------------------------------

def get_mod_name(args: Namespace) -> tuple[Namespace, str]:
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