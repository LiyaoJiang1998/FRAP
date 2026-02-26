from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union


@dataclass
class RunConfig:
    # Guiding text prompt
    prompt: Union[List[str], str] = None
    # Use a text file, each line is a prompt
    prompt_file: Path = None
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    # Which token indices to alter with FRAP
    token_indices: Union[List[List[int]], List[int], str] = "all"
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [0])
    # Path to save all outputs to
    output_path: Path = Path('./outputs')

    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply FRAP
    max_iter_to_alter: int = 25
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or FRAP
    run_standard_sd: bool = False
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False    

    # Configurations
    redo_current_step: bool = False
    # Initial Latent Selection
    num_initial_latents: int = 4
    num_initial_steps: int = 15
    # Alpha Optimization
    scale_factor: float = 1.0 # Scale factor for updating the alphas
    scale_range: tuple = field(default_factory=lambda: (1.0, 1.0)) # Start and end values used for scaling the scale factor - constant scaling
    alpha_init: float = 0 # Initial values of alpha, for token in indices_to_alter.
    alpha_for_phi_one: float = 0 # Initial values of alpha, for rest of the tokens. So PHI=1, i.e. deafult weighting.
    alpha2phi_fcn: Dict = field(default_factory=lambda: {'PHI_UB': 1.4, 'PHI_LB': 0.6})
    loss_info: Dict = field(default_factory=lambda: {'loss_coefficients': [1.0, 1.0]})

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
    