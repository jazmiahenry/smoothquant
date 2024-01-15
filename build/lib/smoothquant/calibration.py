"""
Calibrates model and datset for quantization. 

Parameters:
    model: Huggingface Transformer model input
    dataset: file type json
    tensor: int

Functions:
    get_act_scales
    stat_tensor
    stat_input_hook
    get_static_decoder_layer_scales
    state_io_hook
"""
import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm

def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    """
    Create dict of scales
    Input: 
        model: Transformers model
        tokenizer: Transformers tokenizer
        dataset_path: str
        num_samples: int
        seq_len: int
    Output:
        act_scales: dict
    """
    model.eval()
    device = next(model.parameters()).device # Device types (gpu, cpu)
    act_scales = {}

def stat_tensor(name, tensor):
    """
    Reshape tensors and return detached maximum float value.
    """
    hidden_dim = tensor.shape[-1] # Reshapping Tensor shape
    tensor = tensor.view(-1, hidden_dim).abs().detach() # Returns Tensor detached from current graph
    comming_max = torch.max(tensor, dim=0)[0].float().cpu() # Returns max value of tensor run on cpu
    if name in act_scales:
        act_scales[name] = torch.max(act_scales[name], comming_max)
    else:
        act_scales[name] = comming_max

def stat_input_hook(m, x, y, name):
    """
    Collects hook names and registers forward hook to nn.module.register
    Input: 
        hook: tuple
        dataset: file type json -> object
    Output:
        hooks: list of module names
        dataset: object
    """
    if isinstance(x, tuple):
        x = x[0]
    stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    # Calculate time to tokenize samples
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales

@torch.no_grad()
def get_static_decoder_layer_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    """
    Scale decoder layer 
    Input:
        model: Transformer model
        tokenizer: Transformer tokenizer
        dataset_path: str
        num_samples: int
        seq_len: int
    
    Return:
        act_dict: dict
    """
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

def stat_io_hook(m, x, y, name):
    """
    Save static decoder layers to hook 
    Collect decoder layer information in a dict and save to a list 
    """
    if isinstance(x, tuple):
        x = x[0]
    if name not in act_dict or "input" not in act_dict[name]:
        act_dict[name]["input"] = x.detach().abs().max().item()
    else:
        act_dict[name]["input"] = max(
            act_dict[name]["input"], x.detach().abs().max().item())
    if isinstance(y, tuple):
        y = y[0]
    if name not in act_dict or "output" not in act_dict[name]:
        act_dict[name]["output"] = y.detach().abs().max().item()
    else:
        act_dict[name]["output"] = max(
            act_dict[name]["output"], y.detach().abs().max().item())

    hooks = []
    for name, m in model.named_modules(): # Save hooks to nn.module.register
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(
                partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")

    # Collect activation scales
    pbar = tqdm(range(num_samples)) # Calculate time it takes to collect samples
    dataset = load_dataset('json', data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42) 

    # Tokenize dataset and run inside model
    for i in pbar: 
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                            max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    # Collect decoder layer scales w/ dict of hidden layers in list
    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        scale_dict["attn_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.q_proj"]['input'] / 127
        scale_dict["q_output_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.q_proj"]['output'] / 127
        scale_dict["k_output_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.k_proj"]['output'] / 127
        scale_dict["v_output_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.v_proj"]['output'] / 127
        scale_dict["out_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.self_attn.out_proj"]['input'] / 127
        scale_dict["fc1_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.fc1"]['input'] / 127
        scale_dict["fc2_input_scale"] = act_dict[
            f"model.decoder.layers.{idx}.fc2"]["input"] / 127
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict
