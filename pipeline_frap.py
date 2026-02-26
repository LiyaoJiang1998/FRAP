
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention, aggregate_attention_batched

logger = logging.get_logger(__name__)


class FrapPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using FRAP.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds
    
    
    def _compute_attention_per_index(self,
                                    attention_maps: torch.Tensor,
                                    indices_to_alter: List[List[int]],
                                    smooth_attentions: bool = False,
                                    sigma: float = 0.5,
                                    kernel_size: int = 3,
                                    normalize_eot: bool = False) -> List[torch.Tensor]:        
        list_of_smoothed_maps = []
        # process each prompt in the batch
        for idx, prompt in enumerate(self.prompts):
            if normalize_eot:
                last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
            else:
                last_idx = -1

            attention_for_text = attention_maps[idx, :, :, 1:last_idx]
            attention_for_text *= 100
            attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

            # Shift indices since we removed the first token
            indices_to_alter_local = []
            for i, _ in enumerate(indices_to_alter):
                indices_to_alter_local.append([index - 1 for index in indices_to_alter[i]])

            smoothed_maps_list = []
            for i in indices_to_alter_local[idx]:
                image = attention_for_text[:, :, i]
                if smooth_attentions:
                    smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(self._execution_device)
                    input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                    image = smoothing(input).squeeze(0).squeeze(0)
                smoothed_maps_list.append(image)
                            
            list_of_smoothed_maps.append(smoothed_maps_list)
        
        return list_of_smoothed_maps
    
    
    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   indices_to_alter: List[List[int]],
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False,
                                                   inferenced_with_cfg: bool=False):
        if inferenced_with_cfg:
            attention_maps = aggregate_attention_batched(
                attention_store=attention_store,
                res=attention_res,
                from_where=("up", "down", "mid"),
                is_cross=True,
                batch_size=2*len(indices_to_alter))[len(indices_to_alter):] # take attention_maps from the text conditioned 
        else:
            attention_maps = aggregate_attention_batched(
                attention_store=attention_store,
                res=attention_res,
                from_where=("up", "down", "mid"),
                is_cross=True,
                batch_size=len(indices_to_alter))
        
        list_of_smoothed_maps = self._compute_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        
        return list_of_smoothed_maps
    
    
    @staticmethod
    def _prompt_weighting(prompt_embeds, weightings):
        # By interpolating between prompt_embeds[0] (c_uncond) and prompt_embeds[1] (c_cond)
        c_uncond, c_cond = prompt_embeds.chunk(2)
        c_weighted = weightings * c_cond + (1 - weightings) * c_uncond
        prompt_embeds_weighted = torch.cat([c_uncond, c_weighted], dim=0)
        return prompt_embeds_weighted
    

    @staticmethod
    def _update_alphas_with_loss(alphas: torch.Tensor, loss: torch.Tensor, step_size: float, indices_to_alter: List[List[int]]) -> torch.Tensor:
        """ Update the prompt weighting alphas according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [alphas], retain_graph=False)[0]
        updated_alphas = alphas - step_size * grad_cond
        # Note: only update alphas for indices_to_alter tokens of each prompt, alphas for other tokens should remain unchanged
        alphas = alphas.clone().detach()
        for idx, indices_to_alter_prompt in enumerate(indices_to_alter):
            alphas[idx, indices_to_alter_prompt,:] = updated_alphas[idx, indices_to_alter_prompt,:]
        
        return alphas


    @staticmethod
    def init_alphas(do_classifier_free_guidance, alpha_for_phi_one, batch_size, num_images_per_prompt, prompt_embeds, indices_to_alter, alpha_init):
        # Prepare initial per-token prompt weighting scales (init alphas to alpha_for_phi_one, so phis are 1, i.e. default weighting)
        if do_classifier_free_guidance:
            alphas = alpha_for_phi_one * torch.ones((batch_size * num_images_per_prompt, prompt_embeds[0].shape[0], 1), device=prompt_embeds.device, dtype=prompt_embeds.dtype) # shape: [bs, 77, 1]
            for idx, indices_to_alter_prompt in enumerate(indices_to_alter):
                alphas[idx, indices_to_alter_prompt,:] = alpha_init # init alphas to alpha_init for indices_to_alter tokens
        return alphas


    @staticmethod
    def get_phis_from_alphas(alphas, alpha2phi_fcn):
        PHI_UB = alpha2phi_fcn['PHI_UB']
        PHI_LB = alpha2phi_fcn['PHI_LB']
        # Bound the per-token prompt weighting scales
        phis = PHI_LB + (PHI_UB - PHI_LB) * torch.sigmoid(alphas)
        return phis

    
    @staticmethod
    def compute_loss_initZ(list_of_smoothed_maps: List[List[torch.Tensor]],
                        subject_token_indices = [2,5], # [3,7]
                        subject_token_counts = [1,1],
                        subject_modifiers_indices_list = [[2],[6]],
                        loss_info={'loss_coefficients': [1.0, 1.0]},
                        ) -> torch.Tensor:

        def adjust_indices(subject_token_indices, subject_modifiers_indices_list):
            # Obtain Token Grouping:
            subject_token_indices = [x-1 for x in subject_token_indices] # align index since <bos> is removed
            subject_modifiers_indices_list = [[x-1 for x in subject_modifiers_indices] for subject_modifiers_indices in subject_modifiers_indices_list]
            return subject_token_indices, subject_modifiers_indices_list

        subject_token_indices, subject_modifiers_indices_list = adjust_indices(subject_token_indices, subject_modifiers_indices_list)

        loss_list = []
        for batch_idx, smoothed_maps in enumerate(list_of_smoothed_maps):
            subject_presence_loss_list = []
            bind_loss_list = []
            image = smoothed_maps[0]
            for idx, obj_idx in enumerate(subject_token_indices):
                image = smoothed_maps[obj_idx]

                subject_presence_loss_list.append(max(torch.tensor(0., dtype=image.dtype, device=image.get_device()), 1. - image.max()))
                
                if subject_modifiers_indices_list[idx] != []:
                    for jdx, att_jdx in enumerate(subject_modifiers_indices_list[idx]):
                        color_image = smoothed_maps[att_jdx]

                        image_ = image
                        color_image_ = color_image
                        color_image_ = color_image_ / color_image_.max() * image_.max()
                        image_ = torch.nn.functional.softmax(image_.view(-1),dim=0)
                        color_image_ = torch.nn.functional.softmax(color_image_.view(-1),dim=0)

                        bind_loss_list.append(torch.min(image_, color_image_).mean())

            if len(subject_presence_loss_list) != 0:
                subject_presence_loss = torch.stack(subject_presence_loss_list).mean()
            else:
                subject_presence_loss = torch.tensor(0., dtype=image.dtype, device=image.get_device())
            
            if len(bind_loss_list) != 0:
                bind_loss = torch.stack(bind_loss_list).mean()
            else:
                bind_loss = torch.tensor(0., dtype=image.dtype, device=image.get_device())
            
            loss_list.append(subject_presence_loss*loss_info['loss_coefficients'][0] - bind_loss*loss_info['loss_coefficients'][1])
                
        return torch.stack(loss_list)


    @staticmethod
    def compute_loss_alpha(list_of_smoothed_maps: List[List[torch.Tensor]],
                    subject_token_indices = [2,5], # [3,7]
                    subject_token_counts = [1,1],
                    subject_modifiers_indices_list = [[2],[6]],
                    loss_info={'loss_coefficients': [1.0, 1.0]},
                    ) -> torch.Tensor:

        loss_list = FrapPipeline.compute_loss_initZ(list_of_smoothed_maps,
                                                subject_token_indices,
                                                subject_token_counts,
                                                subject_modifiers_indices_list,
                                                loss_info)

        return loss_list.mean()


    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[List[int]],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
            redo_current_step: bool = False,
            num_initial_latents = 4,
            num_initial_steps = 15,
            scale_factor: int = 1.5,
            scale_range: Tuple[float, float] = (1.0, 1,0),
            alpha_init: float = 0.0,
            alpha_for_phi_one: float = 0.0,
            alpha2phi_fcn: Dict = {'PHI_UB': 1.4, 'PHI_LB': 0.6},
            loss_info: Dict = {'loss_coefficients': [1.0, 1.0]}
        ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """        
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        self.prompts = [x for x in prompt for i in range(num_images_per_prompt)] * num_initial_latents
        indices_to_alter = [x for x in indices_to_alter for i in range(num_images_per_prompt)] * num_initial_latents

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        prompt_embeds_weighted = prompt_embeds

        # 3.1. Analyze prompt
        from utils.text_analyzer import text_analyzer
        analyzer = text_analyzer(self.tokenizer)
        if prompt is not None and isinstance(prompt, str):
            objects_indices, object_counts = analyzer._extract_objects_indices(prompt)
            attribute_indices = analyzer._extract_attribution_indices(prompt, objects_indices)
        elif prompt is not None and isinstance(prompt, list):
            if len(prompt) > 1:
                raise Exception("Only handling a single prompt!")
            objects_indices, object_counts = analyzer._extract_objects_indices(prompt[0])
            attribute_indices = analyzer._extract_attribution_indices(prompt[0], objects_indices)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 6. Prepare latent variables
        if num_initial_latents == 1 or run_standard_sd:
            num_channels_latents = self.unet.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds_weighted.dtype,
                device,
                generator,
                latents,
            )
        else:
            # Initial Latent Code Selection
            num_channels_latents = self.unet.in_channels
            initial_latents = self.prepare_latents(
                batch_size * num_images_per_prompt * num_initial_latents,
                num_channels_latents,
                height,
                width,
                prompt_embeds_weighted.dtype,
                device,
                generator,
                None,
            )
            
            prompt_embeds_initial = prompt_embeds.repeat(1, num_initial_latents, 1).view(prompt_embeds.shape[0]*num_initial_latents,prompt_embeds.shape[1],prompt_embeds.shape[2]).detach().clone()
            latents = initial_latents.detach().clone()
            
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_initial_steps) as progress_bar:            
                for i, t in enumerate(timesteps):

                    if i >= num_initial_steps:
                        progress_bar.update()
                        break
                    
                    # Forward pass
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_initial, cross_attention_kwargs=cross_attention_kwargs).sample
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Get max activation value for each subject token
                    list_of_smoothed_maps = self._aggregate_and_get_max_attention_per_token(
                        attention_store=attention_store,
                        indices_to_alter=indices_to_alter,
                        attention_res=attention_res,
                        smooth_attentions=smooth_attentions,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        normalize_eot=sd_2_1,
                        inferenced_with_cfg=do_classifier_free_guidance)
                        
                    initial_latent_scores_batched = self.compute_loss_initZ(list_of_smoothed_maps=list_of_smoothed_maps, 
                                                                           subject_token_indices=objects_indices, 
                                                                           subject_token_counts=object_counts, 
                                                                           subject_modifiers_indices_list=attribute_indices,
                                                                           loss_info=loss_info)
                    initial_latent_scores_batched = initial_latent_scores_batched.tolist()
                    # print(f'Inital Iteration {i} | initial_latent_scores_batched: {[round(x,4) for x in initial_latent_scores_batched]}')
                    
                    # Find best initial_latent with lowest score
                    top_indices =[]
                    for batch_idx in range(batch_size*num_images_per_prompt):
                        current_prompt_options = initial_latent_scores_batched[batch_idx::batch_size*num_images_per_prompt]
                        best_option_index = current_prompt_options.index(min(current_prompt_options))
                        best_option_index = batch_idx + best_option_index * batch_size*num_images_per_prompt
                        top_indices.append(best_option_index)
                    # print(f'Inital Iteration {i} | top_indices: {[x for x in top_indices]}')
                                    
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
            
            latents = initial_latents[top_indices, :, :, :]
            latents = latents.clone().detach()
        
        # 7. Denoising loop
        self.prompts = self.prompts[:batch_size * num_images_per_prompt]
        indices_to_alter = indices_to_alter[:batch_size * num_images_per_prompt]
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))
        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1
        
        alphas = self.init_alphas(do_classifier_free_guidance, alpha_for_phi_one, batch_size, num_images_per_prompt, prompt_embeds, indices_to_alter, alpha_init)
        
        # Adaptive Prompt Weighting
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):                
                if not run_standard_sd:
                    with torch.enable_grad():
                        if i < max_iter_to_alter:                            
                            # Apply Prompt Weighting before forward
                            alphas = alphas.clone().detach().requires_grad_(True) # shape: [77, 1]
                            phis = self.get_phis_from_alphas(alphas, alpha2phi_fcn)
                            # By interpolating between prompt_embeds[0] (c_uncond) and prompt_embeds[1] (c_cond)
                            prompt_embeds_weighted = self._prompt_weighting(prompt_embeds, phis)
                                        
                            if redo_current_step:
                                # Forward pass of denoising with text conditioning
                                noise_pred_text = self.unet(latents, t,
                                                            encoder_hidden_states=prompt_embeds_weighted[batch_size*num_images_per_prompt:], 
                                                            cross_attention_kwargs=cross_attention_kwargs).sample
                                self.unet.zero_grad()
                            else:    
                                # Forward pass
                                # expand the latents if we are doing classifier free guidance
                                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                                # predict the noise residual
                                noise_pred = self.unet(latent_model_input, t, 
                                                       encoder_hidden_states=prompt_embeds_weighted, 
                                                       cross_attention_kwargs=cross_attention_kwargs).sample
                                self.unet.zero_grad()
                                # perform guidance
                                if do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                
                            list_of_smoothed_maps = self._aggregate_and_get_max_attention_per_token(
                                attention_store=attention_store,
                                indices_to_alter=indices_to_alter,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=sd_2_1,
                                inferenced_with_cfg=not redo_current_step)
                            
                            loss = self.compute_loss_alpha(list_of_smoothed_maps=list_of_smoothed_maps,
                                                                          subject_token_indices=objects_indices, 
                                                                          subject_modifiers_indices_list=attribute_indices,
                                                                          loss_info=loss_info)
                                                        
                            if loss != 0:
                                alphas = self._update_alphas_with_loss(alphas=alphas, loss=loss, 
                                                                    step_size=scale_factor*np.sqrt(scale_range[i]), indices_to_alter=indices_to_alter)
                                # Update phis and prompt_embeds_weighted with new alphas
                                phis = self.get_phis_from_alphas(alphas, alpha2phi_fcn)
                                
                                # for idx, indices_to_alter_prompt in enumerate(indices_to_alter):
                                    # print("Phis-batch_idx%s:"%(idx), [round(x,4) for x in phis[idx, indices_to_alter_prompt,:].flatten().tolist()])
                                
                            # print(f'Iteration {i} | Loss: {loss:0.4f}')
                        else:
                            prompt_embeds_weighted = prompt_embeds
                            
                            if redo_current_step:
                                # Forward pass of denoising with text conditioning
                                noise_pred_text = self.unet(latents, t,
                                                            encoder_hidden_states=prompt_embeds_weighted[batch_size*num_images_per_prompt:], cross_attention_kwargs=cross_attention_kwargs).sample
                                self.unet.zero_grad()
                            else:    
                                # Forward pass
                                # expand the latents if we are doing classifier free guidance
                                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                                # predict the noise residual
                                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_weighted, cross_attention_kwargs=cross_attention_kwargs).sample
                                self.unet.zero_grad()
                                # perform guidance
                                if do_classifier_free_guidance:
                                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                                                                
                if run_standard_sd or redo_current_step:
                    if run_standard_sd:
                        prompt_embeds_weighted = prompt_embeds
                    else:
                        prompt_embeds_weighted = self._prompt_weighting(prompt_embeds, phis)
                    
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_weighted,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds_weighted.dtype)
        image, has_nsfw_concept = image, False

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
