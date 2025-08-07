# from T2IBenchmark.datasets import get_coco_30k_captions, get_coco_fid_stats
# from T2IBenchmark import calculate_fid, calculate_clip_score
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
import argparse
import os
from tqdm import tqdm
import numpy as np
import time
import cv2
from typing import Any, Callable, Dict, List, Optional, Union
from operator import itemgetter
import torch.nn.functional as F

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from RALU_NTDM import NT_DM


class FluxPipeline_RALU(FluxPipeline):
    def set_params(self, use_RALU_default, level, N=None, e=None, up_ratio=0.3):
        if use_RALU_default:
            if level == 4:    
                N = [5, 6, 7] # 4x
                e = [0.3, 0.45, 1.0]
                shift = [5.019, 2.589, 2.226]
                Z = 6.317 # Z = 1/sqrt(c)
                up_ratio = 0.3
                
            elif level == 7:
                N = [2, 3, 5] # 7x
                e = [0.2, 0.3, 1.0]
                shift = [8.141, 2.856, 2.193]
                Z = 6.261
                up_ratio = 0.1
                
            else:
                raise ValueError(f"Invalid level: {level}. Only level=4 or level=7 is supported.")
            
        else:
            assert N is not None and e is not None, "N and e must be provided when use_RALU_default is False."
            shift, Z = NT_DM(N, e, self.scheduler.shift) # Use the shift factor from the scheduler as h_ori
        
        s = [0.0]
        alphas = []
        betas = []
        k = [4, 4]

        for i in range(len(N)-1):
            s.append(e[i] / (Z*(1-e[i]) + e[i]))
            alphas.append(1-s[i+1])
            betas.append(1 / (Z*(1-e[i]) + e[i]))
        
        self.params = {
            "N": N,
            "shift": shift,
            "e": e,
            "Z": Z,
            "up_ratio": up_ratio,
            "s": s,
            "alphas": alphas,
            "betas": betas,
            "k": k,
        }
    
    def generate_noise(self, device, height, width):
        M = (height//16) * (width//16)
        f0 = 2
        
        batch_size = 1
        latent_dim = 64
        up_ratio = self.params["up_ratio"]
        Z = self.params["Z"]

        M_down = M // (f0 ** 2)
        num_latents_stage2 = 4 * int(M_down * up_ratio) + (M_down - int(M_down * up_ratio))
        num_latents_stage3 = M
        num_latents = [num_latents_stage2, num_latents_stage3] 
        
        a_stage2 = M_down - int(M_down * up_ratio)

        configs_stage2 = [
            {'start': 0, 'end': a_stage2, 'block_size': 1, 'gamma': 0},
            {'start': a_stage2, 'end': num_latents_stage2, 'block_size': f0*f0, 'gamma': -1/(Z**2)+1e-6},
        ]
        
        a_stage3 = f0 * f0 * (M_down - int(M_down * up_ratio))
        configs_stage3 = [
            {'start': 0, 'end': a_stage3, 'block_size': f0*f0, 'gamma': -1/(Z**2)+1e-6},
            {'start': a_stage3, 'end': num_latents_stage3, 'block_size': 1, 'gamma': 0},
        ]
        
        configs = (configs_stage2, configs_stage3)
        n_primes = []
        for i in range(len(configs)):
            config = configs[i]
            blocks = []
            for cfg in config:
                n = cfg['end'] - cfg['start']
                block_size = cfg['block_size']
                gamma = cfg['gamma']
                num_blocks = n // block_size
                rem = n % block_size

                # block diagonal part
                if block_size > 1:
                    Sigma_block = torch.eye(block_size) + gamma * torch.ones((block_size, block_size))
                    blocks.extend([Sigma_block] * num_blocks)
                    if rem > 0:
                        blocks.append(torch.eye(rem))
                else:
                    # block_size==1: I
                    blocks.extend([torch.eye(1)] * n)

            Sigma_pixel = torch.block_diag(*blocks).to(device=device)  # (num_latents, num_latents)

            n_prime = torch.stack([
                torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=torch.zeros(num_latents[i], device=device),
                    covariance_matrix=Sigma_pixel
                ).sample((batch_size,))
                for _ in range(latent_dim)
            ], dim=-1)  # shape: (batch_size, num_latents, latent_dim)
            n_primes.append(n_prime)
        
        self.n_primes = n_primes

    @staticmethod
    def _upsample_latent_ids(latent_ids, offset):
        offsets = torch.tensor([0.0, offset], dtype=torch.float32).to(latent_ids.device)
        grid_b, grid_c = torch.meshgrid(offsets, offsets, indexing="ij")
        candidate_offsets = torch.stack([grid_b, grid_c], dim=-1) 

        first_col = latent_ids[:, :1]
        rest = latent_ids[:, 1:] 
        
        rest_expanded = rest.unsqueeze(1).unsqueeze(1) 
        cand_offsets = candidate_offsets.unsqueeze(0)

        modified_rest = rest_expanded + cand_offsets 
        first_col_expanded = first_col.unsqueeze(1).unsqueeze(1)
        first_col_expanded = first_col_expanded.expand(-1, 2, 2, -1)

        new_rows = torch.cat([first_col_expanded, modified_rest], dim=-1)
        result = new_rows.reshape(-1, 3)

        return result

    @staticmethod
    def _sort_latents_and_ids(latents, latent_ids):
        _, indices_col2 = torch.sort(latent_ids[:, 2], stable=True)
        latent_ids_sorted_by_col2 = latent_ids[indices_col2]

        _, indices_col1 = torch.sort(latent_ids_sorted_by_col2[:, 1], stable=True)
        final_indices = indices_col2[indices_col1]

        sorted_ids = latent_ids[final_indices]
        sorted_latents = latents[:, final_indices, :]
        
        return sorted_latents, sorted_ids

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None, desc=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )
        config = self._progress_bar_config.copy()
        if desc is not None:
            config["desc"] = desc

        if iterable is not None:
            return tqdm(iterable, **config)
        elif total is not None:
            return tqdm(total=total, **config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")
    
    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
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
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        self.enable_vae_tiling()
        self.vae.to('cuda')
        
        # accel params
        # N = self.params["N"]
        # shift = self.params["shift"]
        # e = self.params["e"]
        # Z = self.params["Z"]
        # up_ratio = self.params["up_ratio"]
        # s = self.params["s"]
        # alphas = self.params["alphas"]
        # betas = self.params["betas"]
        # k = self.params["k"]
        N, shift, e, Z, up_ratio, s, alphas, betas, k = itemgetter(
            "N", "shift", "e", "Z", "up_ratio", "s", "alphas", "betas", "k"
        )(self.params)

        start_time = time.time()
        
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        # has_neg_prompt = negative_prompt is not None or (
        #     negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        # )

        # do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        # if do_true_cfg:
        #     (
        #         negative_prompt_embeds,
        #         negative_pooled_prompt_embeds,
        #         _,
        #     ) = self.encode_prompt(
        #         prompt=negative_prompt,
        #         prompt_2=negative_prompt_2,
        #         prompt_embeds=negative_prompt_embeds,
        #         pooled_prompt_embeds=negative_pooled_prompt_embeds,
        #         device=device,
        #         num_images_per_prompt=num_images_per_prompt,
        #         max_sequence_length=max_sequence_length,
        #         lora_scale=lora_scale,
        #     )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        total_timesteps = sum(N)

        for idx in range(len(N)):
            S = 1 - s[idx]
            E = 1 - e[idx]
            s_prime = float(S / (shift[idx] - (shift[idx]-1) * S))
            e_prime = float(E / (shift[idx] - (shift[idx]-1) * E))
            sigmas = np.linspace(s_prime, e_prime, N[idx] + 1)

            # a) tailored scheduler shifting
            timesteps = shift[idx] * sigmas / (1 + (shift[idx] - 1) * sigmas)
            timesteps = torch.from_numpy(timesteps * 1000).to(device).to(torch.float32)
            
            num_inference_steps = len(timesteps)

            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
                negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
            ):
                negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
                negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
            ):
                ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {}

            image_embeds = None
            negative_image_embeds = None
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                image_embeds = self.prepare_ip_adapter_image_embeds(
                    ip_adapter_image,
                    ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                )
            if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
                negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                    negative_ip_adapter_image,
                    negative_ip_adapter_image_embeds,
                    device,
                    batch_size * num_images_per_prompt,
                )

            ###########################################################################################################
            f0 = 2

            # b) upsampling or downsampling
            # Stage 1
            if idx == 0:
                C, M, D = latents.shape
                M_down = M // (f0 ** 2)
                
                latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
                latents = F.interpolate(latents, size=(latents.shape[-2] // f0, latents.shape[-1] // f0), mode='nearest')
                latents = self._pack_latents(latents, batch_size, num_channels_latents, height // (8*f0), width // (f0*8))

                latent_image_ids = self._prepare_latent_image_ids(batch_size, height // (16*f0), width // (16*f0), device, dtype=latent_image_ids.dtype)
            
            # Stage 2
            elif idx == 1:
                # Tweedie
                latents_x0 = (latents.to(torch.float32) - noise_preds[-1].to(torch.float32) * noise_pred_timestep[-1].to(torch.float32) * 1e-3).to(latents_dtype)
                latents_x0 = self._unpack_latents(latents_x0, height // f0, width // f0, self.vae_scale_factor)

                # decoding
                latents_x0 = (latents_x0 / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                latents_x0 = self.vae._decode(latents_x0).sample

                # edge detection
                latents_x0 = latents_x0.squeeze().permute(1,2,0).to(torch.float32).detach().cpu().numpy()
                gray_x0 = 0.299 * latents_x0[:,:,0] + 0.587 * latents_x0[:,:,1] + 0.114 * latents_x0[:,:,2]
                gray_x0 = (np.clip(gray_x0, 0, 1) * 255).astype(np.uint8)
                edges = torch.tensor(cv2.Canny(gray_x0, threshold1=100, threshold2=200))
                
                edges = (
                    edges.view(edges.shape[0]//16, 16, edges.shape[1]//16, 16)
                        .permute(0, 2, 1, 3).reshape(edges.shape[0]//16, edges.shape[1]//16, -1).sum(dim=-1).flatten().float()
                )
                edges += torch.randn(edges.size()) * 1e-6

                # find top k patches to upsample
                topk = int(M_down * up_ratio)
                top_values, top_indices = torch.topk(edges, topk, largest=True)

                indices_all = torch.arange(latents.size(1), device=latents.device)
                mask = torch.ones(latents.size(1), dtype=torch.bool, device=latents.device)
                mask[top_indices] = False
                indices_1x = indices_all[mask]
                indices_2x, _ = torch.sort(top_indices)

                latents_1x = latents[:, indices_1x, :]
                latents_2x = latents[:, indices_2x, :]
                
                latent_image_ids_1x = latent_image_ids[indices_1x, :]
                latent_image_ids_2x = latent_image_ids[indices_2x, :]

                latents_2x = F.interpolate(latents_2x.transpose(1,2), size=latents_2x.size(1)*f0*f0, mode='nearest').transpose(1,2)
                latent_image_ids_2x = self._upsample_latent_ids(latent_image_ids_2x, 0.5)
                
                latents = torch.cat([latents_1x, latents_2x], axis=1)
                latent_image_ids = torch.cat([latent_image_ids_1x, latent_image_ids_2x], axis=0)

            # Stage 3 (upsampling 1x latents, latent_ids)
            else: # idx == 2
                latents_1x = latents[:, :len(indices_1x), :]
                latents_2x = latents[:, len(indices_1x):, :]
                
                latent_image_ids_1x = latent_image_ids[:len(indices_1x), :]
                latent_image_ids_2x = latent_image_ids[len(indices_1x):, :]
                
                latents_1x = F.interpolate(latents_1x.transpose(1,2), size=latents_1x.size(1)*(f0**2), mode='nearest').transpose(1,2)
                latent_image_ids_1x = self._upsample_latent_ids(latent_image_ids_1x, 0.5)
                
                latents = torch.cat([latents_1x, latents_2x], axis=1)
                latent_image_ids = torch.cat([latent_image_ids_1x, latent_image_ids_2x], axis=0) * f0

            # c) add noise
            if idx > 0:
                alpha, beta = alphas[idx-1], betas[idx-1]
                latents = beta * latents + alpha * self.n_primes[idx-1].to(device, latents.dtype)
                if idx == 2:
                    # sorting latents and ids (Stage 3)
                    latents, latent_image_ids = self._sort_latents_and_ids(latents, latent_image_ids)
                
            # d) denoising loop       
            noise_pred_timestep = []
            start = 0
            end = N[idx]

            noise_preds = []
            with self.progress_bar(total=end-start, desc=f"Stage {idx+1}") as progress_bar:
                for i in range(start, end):
                    t = timesteps[i]
                    if self.interrupt:
                        continue
                    
                    if image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_preds.append(noise_pred)
                    noise_pred_timestep.append(timesteps[i+1])
                    # if do_true_cfg:
                    #     if negative_image_embeds is not None:
                    #         self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    #     neg_noise_pred = self.transformer(
                    #         hidden_states=latents,
                    #         timestep=timestep / 1000,
                    #         guidance=guidance,
                    #         pooled_projections=negative_pooled_prompt_embeds,
                    #         encoder_hidden_states=negative_prompt_embeds,
                    #         txt_ids=text_ids,
                    #         img_ids=latent_image_ids,
                    #         joint_attention_kwargs=self.joint_attention_kwargs,
                    #         return_dict=False,
                    #     )[0]
                    #     noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = (latents.to(torch.float32) + noise_pred.to(torch.float32) * (timesteps[i+1] - timesteps[i]).to(torch.float32) * 1e-3).to(latents_dtype)
                    
                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        end_time = time.time()
        latency = end_time - start_time
        # print(f"Latency: {latency:.6f} seconds")

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)