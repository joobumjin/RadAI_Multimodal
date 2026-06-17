import os

from transformers import AutoModel, AutoTokenizer
import torch

def get_deepseek(model_name = 'deepseek-ai/DeepSeek-OCR-2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)
    return model, tokenizer

def infer_img(model, tokenizer, image_file, output_path):
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 768, crop_mode=True, save_results = True)

    return res

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
