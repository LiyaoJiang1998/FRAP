import pprint
from typing import List, Union
import time
from datetime import timedelta
import numpy as np
import json
    
import pyrallis
import torch
from PIL import Image

from config_frap import RunConfig
from pipeline_frap import FrapPipeline

from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-5"
    
    dtype = torch.float16
    stable = FrapPipeline.from_pretrained(stable_diffusion_version, torch_dtype=dtype).to(device)
    
    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: Union[str, List[str]],
                  model,
                  controller: AttentionStore,
                  token_indices: Union[List[int], List[List[int]]],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    redo_current_step=config.redo_current_step,
                    num_initial_latents=config.num_initial_latents,
                    num_initial_steps=config.num_initial_steps,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    alpha_init = config.alpha_init,
                    alpha_for_phi_one=config.alpha_for_phi_one,
                    alpha2phi_fcn=config.alpha2phi_fcn,
                    loss_info=config.loss_info)
    image = outputs.images
    
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    
    if config.prompt_file is not None:
        with open(config.prompt_file) as f:
            prompts = [x.rstrip() for x in f]
    elif isinstance(config.prompt, str):
        prompts = [config.prompt]
    elif isinstance(config.prompt, list):
        prompts = config.prompt
    else:
        raise ValueError("Invalid input prompts format")
    
    token_indices = []
    if isinstance(config.token_indices, str) and config.token_indices=="all":
        # Use all tokens
        for idx, prompt in enumerate(prompts):
            token_indices.append(list(range(1, len(stable.tokenizer(prompt)['input_ids'])-1)))
    elif config.token_indices is None and not config.run_standard_sd:
        # Get input from user
        for idx, prompt in enumerate(prompts):
            token_indices = get_indices_to_alter(stable, prompt)
    else:
        if isinstance(config.token_indices, list) and isinstance(config.token_indices[0], int): 
            token_indices = [config.token_indices for prompt in prompts]
        else:
            # Use the provided list of token_indices
            token_indices = config.token_indices

    images = [[] for x in prompts]
    batch_size = 1
    
    latency_list = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        for i in range(0, len(prompts), batch_size):
            start_time = time.perf_counter()
            
            print(f"Prompt: {i}")
            batch_prompts = prompts[i:i+batch_size]
            batch_token_indices = token_indices[i:i+batch_size]
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(prompt=batch_prompts,
                                    model=stable,
                                    controller=controller,
                                    token_indices=batch_token_indices,
                                    seed=g,
                                    config=config)
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            latency_list.append(elapsed_time)
            print("Latency: %.2fs"%elapsed_time)

            num_images_per_prompt = int(len(image) / len(batch_prompts))
            for idx, prompt in enumerate(batch_prompts):
                prompt = prompt[:250] # filename length limit
                prompt_output_path = config.output_path / prompt
                prompt_output_path.mkdir(exist_ok=True, parents=True)
                prompt_image = image[idx*num_images_per_prompt:(idx+1)*num_images_per_prompt]
                prompt_image = vis_utils.get_image_grid(prompt_image)
                
                prompt_image.save(prompt_output_path / f'{seed}.png')
                images[i+idx].append(prompt_image)
    
    # Save inference latency stats
    latency_array = np.array(latency_list)
    latency_mean = latency_array.mean()
    latency_std = latency_array.std()
    latency_min = latency_array.min()
    latency_max = latency_array.max()
    latency_len = len(latency_array)
    latency_sum = latency_array.sum()
    
    latency_results = {
        'latency_mean': latency_mean,
        'latency_std': latency_std,
        'latency_min': latency_min,
        'latency_max': latency_max,
        'latency_len': latency_len,
        'latency_sum': latency_sum,
        'latency_list': latency_list
    }

    with open(config.output_path / "latency_metrics.json", 'w') as f:
        json.dump(latency_results, f, sort_keys=False, indent=4)
        
    print("Run Total Elapsed Time:", timedelta(seconds=latency_sum))
    
    # save a grid of results across all seeds
    if isinstance(prompts, list):
        for idx, prompt in enumerate(prompts):
            prompt = prompt[:250] # filename length limit
            joined_image = vis_utils.get_image_grid(images[idx])
            joined_image.save(config.output_path / f'{prompt}.png')

if __name__ == '__main__':
    main()
