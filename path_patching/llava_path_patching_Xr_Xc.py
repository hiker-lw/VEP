import yaml
import pickle
import json
import logging
import random
import os
import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
import plotly
import plotly.express as px
random.seed(2024)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

from llava_utils import retrieve_logit_llava_last_token_path_patching

def show_pp(
    m,
    xlabel="Head",
    ylabel="Layer",
    title="", 
    bartitle="Logit Change (%)",
    animate_axis=None,
    highlight_points=None,
    highlight_name="",
    return_fig=False,
    show_fig=False,
    save_fig_path="",
    **kwargs,
):
    """
    Plot a heatmap of the values in the matrix `m`
    """

    if animate_axis is None:
        fig = px.imshow(
            m,
            title=title if title else "",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )
    else:
        if m.ndim == 3:
            m = einops.rearrange(m, "a b c -> a c b")
        else:
            raise ValueError("Matrix 'm' must be 3D when 'animate_axis' is provided.")
        
        fig = px.imshow(
            m,
            title=title if title else "",
            animation_frame=animate_axis,
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            **kwargs,
        )

    fig.update_layout(
        coloraxis=dict(colorbar=dict(
            title=bartitle,
            thicknessmode="pixels",
            thickness=50,
            lenmode="pixels",
            len=300,
            yanchor="top",
            y=1,
            ticks="outside",
        )),
    )

    if highlight_points is not None:
        if len(highlight_points) != 2 or len(highlight_points[0]) != len(highlight_points[1]):
            raise ValueError("highlight_points must be a tuple/list of two arrays of the same length.")
        
        fig.add_scatter(
            x=highlight_points[1],
            y=highlight_points[0],
            mode="markers",
            marker=dict(color="green", size=10, opacity=0.5),
            name=highlight_name,
        )

    fig.update_layout(
        yaxis_title=ylabel,
        xaxis_title=xlabel,
        xaxis_range=[-0.5, m.shape[1] - 0.5],
        showlegend=highlight_points is not None,
        legend=dict(x=-0.1) if highlight_points is not None else None,
    )

    if highlight_points is not None:
        fig.update_yaxes(range=[m.shape[0] - 0.5, -0.5], autorange=False)

    if show_fig:
        fig.show()
    if save_fig_path:
        fig.write_image(save_fig_path, scale=3)
    if return_fig:
        return fig

def extract_answer_pope(text):
    # Only keep the first sentence
    if text.find('.') != -1:
        text = text.split('.')[0]

    # Remove commas for easier processing
    text = text.replace(',', '')
    words = text.split(' ')
    # Determine the answer based on the presence of negation words
    if 'No' in words or 'not' in words or 'no' in words:
        return "no"
    else:
        return "yes"

def add_pre_hook(module, name, cache, pos=None, read=False):
    """Add a hook to the module to save and replace the module inputs.

    Parameters
    ----------
    module : nn.Module
        The module whose inputs need to be replaced and saved.
    name : str
        The name used when saving the inputs in the cache.
    cache : Dict
        The cache used to store the inputs.
    pos : int, optional
        The position of the token to save/replace in the sentence, by default None.
    read : bool, optional
        Whether to save or replace. If True, save the inputs; by default False.
    """
    def read_hook(module, input):
        if pos is None:
            cache[name] = input
        else:
            cache[name] = input[0][:, pos]
        return input

    def write_hook(module, input):
        if pos is None:
            return cache[name]    
        input[0][:, pos] = cache[name]
        return input

    if read:
        return module.register_forward_pre_hook(read_hook)
    else:
        return module.register_forward_pre_hook(write_hook)
    
def add_pre_hook_single_head(module, name, cache, pos, read=False, head_idx=None):
    """Parameters are the same as the previous function, except that this function can replace the results corresponding to a single attention head only."""

    def read_hook(module, input):
        # [batch, seq_len, dim]
        if pos is None:
            cache[name] = input
        else:
            cache[name] = input[0][:, pos]
        return input

    def write_hook(module, input):
        input[0][:, pos, head_idx * head_dim : (head_idx + 1) * head_dim] = cache[name][:, head_idx * head_dim : (head_idx + 1) * head_dim]
        return input

    if read:
        return module.register_forward_pre_hook(read_hook)
    else:
        return module.register_forward_pre_hook(write_hook)    

@torch.no_grad()
def path_patching(args_Xr, args_Xc, tokenizer, model, image_processor, llava_model_name, probing_token="No"):
    prob_token_index = tokenizer.encode(probing_token)[1]
    yes_token_index = tokenizer.encode("Yes")[1]
    no_token_index = tokenizer.encode("No")[1]
    # Initialize a matrix to record the results
    results = torch.zeros(size=(num_layers, num_attention_heads))

    # Caches for the model internal representations corresponding to Xr and Xc
    Hr = {}
    Hc = {}

    # Forward A: record the activations for the Xr input
    hooks = []
    for i in range(num_layers):
        for head_in_name in head_in_name_list:
            name = head_in_name.format(i=i)
            module = model.get_submodule(name)  # Retrieve the nn.Module to hook by name
            hook = add_pre_hook(
                module,
                name,
                Hr,
                -1,
                True,
            )
            hooks.append(hook)

    first_token_generated_probs = retrieve_logit_llava_last_token_path_patching(args_Xr, tokenizer, model, image_processor, llava_model_name)
    prob_token_probs_ratio = first_token_generated_probs[0, prob_token_index] / (first_token_generated_probs[0, yes_token_index] + first_token_generated_probs[0, no_token_index])
    # print(f"prob_token_probs_ratio: {prob_token_probs_ratio}")
    for hook in hooks:
        hook.remove()
   
    # Forward B: record the activations for the Xc input
    hooks = []
    for i in range(num_layers):
        name = head_out_name.format(i=i)
        module = model.get_submodule(name)
        hook = add_pre_hook_single_head(
            module,
            name,
            Hc,
            -1,
            True,
        )
        hooks.append(hook)
    
    _ = retrieve_logit_llava_last_token_path_patching(args_Xc, tokenizer, model, image_processor, llava_model_name)[0, prob_token_index]
    for hook in hooks:
        hook.remove()

    # Iterate over all attention heads in all layers
    for source_layer in tqdm.tqdm(range(num_layers)):
        for source_head_idx in list(range(num_attention_heads)):
            # Forward B: during the forward pass, keep other attention heads fixed and replace only the target attention head
            hooks = []
            for j in range(source_layer+1, num_layers): # Fix the attention-head inputs in the layers after the layer containing the target attention head
                for head_in_name in head_in_name_list:
                    name = head_in_name.format(i=j)
                    module = model.get_submodule(name)
                    hook = add_pre_hook(
                        module,
                        name,
                        Hr,
                        -1,
                        False,
                    )
                    hooks.append(hook)

            name = head_out_name.format(i=source_layer)
            module = model.get_submodule(name)
            hook = add_pre_hook_single_head(
                module,
                name,
                Hc,
                -1,
                False,
                source_head_idx
            )
            hooks.append(hook)

            first_token_generated_probs_curr = retrieve_logit_llava_last_token_path_patching(args_Xr, tokenizer, model, image_processor, llava_model_name)
            prob_token_probs_ratio_curr = first_token_generated_probs_curr[0, prob_token_index] / (first_token_generated_probs_curr[0, yes_token_index] + first_token_generated_probs_curr[0, no_token_index])
            print(f"prob_token_probs_ratio_curr: {prob_token_probs_ratio_curr}, prob_token_probs_ratio: {prob_token_probs_ratio}, \
                  relative_ratio: {(prob_token_probs_ratio_curr - prob_token_probs_ratio)/prob_token_probs_ratio * 100}")
            for hook in hooks:
                hook.remove()
            results[source_layer, source_head_idx] = (prob_token_probs_ratio_curr - prob_token_probs_ratio)/prob_token_probs_ratio * 100

    return results

head_in_name_list = ["model.layers.{i}.self_attn.q_proj", "model.layers.{i}.self_attn.k_proj", "model.layers.{i}.self_attn.v_proj"]
head_out_name = "model.layers.{i}.self_attn.o_proj"


Xc_prompt = "Is this image from an outdoor setting?"

if __name__ == "__main__":
    # Randomly select n cases from POPE COCO-Adversarial for analysis
    gt_path = "../LVLM_hallucination/COCO_hallucination_data/coco_pope_adversarial.json"
    response_path = "./llava1.5_7b_response_object_adversarial_without_context.json"

    # Load the ground-truth data
    dataset = {}
    question_id_to_label = {}
    with open(gt_path, "r") as f:
        for line in f:
            item = json.loads(line)
            dataset[item["question_id"]] = item
            question_id_to_label[item["question_id"]] = item["label"]

    # Load the model predictions
    response_dict = {}
    with open(response_path, "r") as f:
        response = json.load(f)
    for item in response:
        response_dict[item["question_id"]] = extract_answer_pope(item["response"])

    analysis_question_id_list = []
    
    for question_id in response_dict.keys():
        if question_id_to_label[question_id] =="yes" and response_dict[question_id] == question_id_to_label[question_id]:
            analysis_question_id_list.append(question_id)
    
    # Initialize the model
    model_path = "../LVLM_hallucination/cache/llava-v1.5-7b"
    llava_model_name = get_model_name_from_path(model_path)
    model_base = None
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base, llava_model_name)

    # Get the number of model layers, number of attention heads, dimensions, etc.
    num_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    results_dict = {}

    for question_id in tqdm.tqdm(random.sample(analysis_question_id_list, 10)):
        prompt = dataset[question_id]["text"]
        image_path = os.path.join("../LVLM_hallucination/COCO_hallucination_data/COCO/val2014", dataset[question_id]["image"])
        args_Xr = type('Args', (), {
            "model_base": None,
            "model_name": llava_model_name,
            "query": prompt,
            "conv_mode": None,
            "image_file": image_path,
            "sep": ",",
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 1,
        })()

        args_Xc = type('Args', (), {
        "model_base": None,
        "model_name": llava_model_name,
        "query": Xc_prompt,
        "conv_mode": None,
        "image_file": image_path,
        "sep": ",",
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 1,
        })()
        
        results = path_patching(args_Xr, args_Xc, tokenizer, model, image_processor, llava_model_name, probing_token="Yes")
        results_dict[question_id] = {}
        results_dict[question_id]["path_patching_results"] = results
        results_dict[question_id]["Xr"] = prompt
        results_dict[question_id]["Xc"] = Xc_prompt
        results_dict[question_id]["image_path"] = image_path
    
        with open("./path_patching/Xr_Xc_path_patching_results_pope_non_hallucination_random_10_samples_label_yes_temp_0.pkl", "wb") as f:
            pickle.dump(results_dict, f)
    
    with open("./path_patching/Xr_Xc_path_patching_results_pope_non_hallucination_random_10_samples_label_yes_temp_0.pkl", "rb") as f:
            results_dict = pickle.load(f)

    stacked_results = torch.stack([item["path_patching_results"] for item in results_dict.values()]) # Stack into a tensor with shape (N, 32, 32)
    average_results = torch.mean(stacked_results, dim=0)
    show_pp(average_results, save_fig_path="./path_patching/Xr_Xc_path_patching_results_pope_non_hallucination_random_10_samples_label_yes_temp_0.png")