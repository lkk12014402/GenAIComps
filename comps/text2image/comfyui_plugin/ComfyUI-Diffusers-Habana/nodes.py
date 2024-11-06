from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
)
from optimum.habana.utils import set_seed

import copy
import os
import torch
from safetensors.torch import load_file
from .utils import SCHEDULERS, token_auto_concat_embeds, vae_pt_to_vae_diffuser, convert_images_to_tensors, convert_tensors_to_images, resize_images
from comfy.model_management import get_torch_device
import folder_paths
# from streamdiffusion import StreamDiffusion
# from streamdiffusion.image_utils import postprocess_image
# from diffusers import StableDiffusionPipeline, AutoencoderKL, AutoencoderTiny
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from diffusers import AutoencoderKL, AutoencoderTiny
from optimum.habana.diffusers import (
    GaudiDDIMScheduler,
    GaudiEulerAncestralDiscreteScheduler,
    GaudiEulerDiscreteScheduler,
)


class DiffusersPipelineLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.bfloat16
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ), }}

    RETURN_TYPES = ("PIPELINE", "AUTOENCODER", "SCHEDULER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "Diffusers"

    def create_pipeline(self, ckpt_name):
        ckpt_cache_path = os.path.join(self.tmp_dir, ckpt_name)
        GaudiStableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=folder_paths.get_full_path("checkpoints", ckpt_name),
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
        ).save_pretrained(ckpt_cache_path, safe_serialization=True)

        kwargs = {'use_habana': True, 'use_hpu_graphs': True, 'gaudi_config': 'Habana/stable-diffusion'}
        
        pipe = GaudiStableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt_cache_path,
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
            **kwargs
        )
        return ((pipe, ckpt_cache_path), pipe.vae, pipe.scheduler)

class DiffusersVaeLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.bfloat16
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (folder_paths.get_filename_list("vae"), ), }}

    RETURN_TYPES = ("AUTOENCODER",)

    FUNCTION = "create_pipeline"

    CATEGORY = "Diffusers"

    def create_pipeline(self, vae_name):
        ckpt_cache_path = os.path.join(self.tmp_dir, vae_name)
        vae_pt_to_vae_diffuser(folder_paths.get_full_path("vae", vae_name), ckpt_cache_path)

        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path=ckpt_cache_path,
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
        )
        
        return (vae,)

class DiffusersSchedulerLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.bfloat16
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE", ),
                "scheduler_name": (list(SCHEDULERS.keys()), ), 
            }
        }

    RETURN_TYPES = ("SCHEDULER",)

    FUNCTION = "load_scheduler"

    CATEGORY = "Diffusers"

    def load_scheduler(self, pipeline, scheduler_name):
        scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=pipeline[1],
            torch_dtype=self.dtype,
            cache_dir=self.tmp_dir,
            subfolder='scheduler'
        )
        return (scheduler,)

class DiffusersModelMakeup:
    def __init__(self):
        self.torch_device = get_torch_device()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE", ), 
                "scheduler": ("SCHEDULER", ),
                "autoencoder": ("AUTOENCODER", ),
            }, 
        }

    RETURN_TYPES = ("MAKED_PIPELINE",)

    FUNCTION = "makeup_pipeline"

    CATEGORY = "Diffusers"

    def makeup_pipeline(self, pipeline, scheduler, autoencoder):
        pipeline = pipeline[0]
        pipeline.vae = autoencoder
        pipeline.scheduler = scheduler
        pipeline.safety_checker = None if pipeline.safety_checker is None else lambda images, **kwargs: (images, [False])
        pipeline.enable_attention_slicing()
        pipeline = pipeline.to(self.torch_device)
        return (pipeline,)

class DiffusersClipTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "maked_pipeline": ("MAKED_PIPELINE", ),
            "positive": ("STRING", {"multiline": True}),
            "negative": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("EMBEDS", "EMBEDS", "STRING", "STRING", )
    RETURN_NAMES = ("positive_embeds", "negative_embeds", "positive", "negative", )

    FUNCTION = "concat_embeds"

    CATEGORY = "Diffusers"

    def concat_embeds(self, maked_pipeline, positive, negative):
        positive_embeds, negative_embeds = token_auto_concat_embeds(maked_pipeline, positive,negative)

        return (positive_embeds, negative_embeds, positive, negative, )

class DiffusersSampler:
    def __init__(self):
        self.torch_device = get_torch_device()
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "maked_pipeline": ("MAKED_PIPELINE", ),
            "positive_embeds": ("EMBEDS", ),
            "negative_embeds": ("EMBEDS", ),
            "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Diffusers"

    def sample(self, maked_pipeline, positive_embeds, negative_embeds, height, width, steps, cfg, seed):
        images = maked_pipeline(
            prompt_embeds=positive_embeds,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            negative_prompt_embeds=negative_embeds,
            # generator=torch.Generator(self.torch_device).manual_seed(seed)
            generator=None
        ).images
        return (convert_images_to_tensors(images),)



def get_dir_list_(folder_name: str):
    folder_name = folder_paths.map_legacy(folder_name)
    output_list = set()
    folders = folder_paths.folder_names_and_paths[folder_name]
    output_folders = []
    for x in folders[0]:
        folders_tmp = folder_paths.recursive_search(x, excluded_dir_names=[".git"])[1]
        output_folders.extend(sorted(list(folders_tmp)))
    return output_folders


class DiffusersModelPath:
    def __init__(self):
        self.dtype = torch.bfloat16

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (get_dir_list_("checkpoints"), ), }}

    RETURN_TYPES = ("MODEL_PATH",)

    FUNCTION = "create_pipeline"

    CATEGORY = "Diffusers"

    def create_pipeline(self, ckpt_name):
        return (ckpt_name,)


SCHEDULER_NAMES = ["euler_ancestral_discrete", "euler_discrete", "ddim"]
class DiffusersPipelineGaudi:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL_PATH", {"tooltip": "The model used for denoising the input latent."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
            "scheduler": (SCHEDULER_NAMES, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
            "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
            "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Diffusers"

    def sample(self, model, seed, steps, cfg, scheduler, positive, negative=None,
            width=512, height=512):

        print(model)
        print(positive)
        print(negative)

        # Set the scheduler
        kwargs = {"timestep_spacing": "linspace", "rescale_betas_zero_snr": False}
        if scheduler == "euler_discrete":
            scheduler = GaudiEulerDiscreteScheduler.from_pretrained(
                model, subfolder="scheduler", **kwargs
            )
        elif scheduler == "euler_ancestral_discrete":
            scheduler = GaudiEulerAncestralDiscreteScheduler.from_pretrained(
                model, subfolder="scheduler", **kwargs
            )
        elif scheduler == "ddim":
            scheduler = GaudiDDIMScheduler.from_pretrained(model, subfolder="scheduler", **kwargs)
        else:
            scheduler = None

        # Set pipeline class instantiation options
        kwargs = {'use_habana': True, 'use_hpu_graphs': True, 'gaudi_config': 'Habana/stable-diffusion'}

        if scheduler is not None:
            kwargs["scheduler"] = scheduler

        kwargs["torch_dtype"] = torch.bfloat16

        # Set pipeline call options
        kwargs_call = {
            "num_images_per_prompt": 1,
            "batch_size": 1,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "eta": 0.0,
            "output_type": "pil",
            "profiling_warmup_steps": 0,
            "profiling_steps": 0,
        }

        if width > 0 and height > 0:
            kwargs_call["width"] = width
            kwargs_call["height"] = height


        kwargs_call["generator"] = None

        kwargs_call["negative_prompt"] = negative

        pipeline = GaudiStableDiffusionPipeline.from_pretrained(
            model,
            **kwargs,
        )

        """
        # for lora
        if args.unet_adapter_name_or_path is not None:
            from peft import PeftModel

            pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args.unet_adapter_name_or_path)
            pipeline.unet = pipeline.unet.merge_and_unload()

        if args.text_encoder_adapter_name_or_path is not None:
            from peft import PeftModel

            pipeline.text_encoder = PeftModel.from_pretrained(
                pipeline.text_encoder, args.text_encoder_adapter_name_or_path
            )
            pipeline.text_encoder = pipeline.text_encoder.merge_and_unload()
        """

        # Set RNG seed
        # set_seed(seed)

        images = pipeline(prompt=positive, **kwargs_call).images

        return (convert_images_to_tensors(images),)


import requests
from requests.exceptions import RequestException
import json
import base64
from io import BytesIO
from PIL import Image

SCHEDULER_NAMES = ["euler_ancestral_discrete", "euler_discrete", "ddim"]
class DiffusersPipelineEndpointGaudi:
    def __init__(self):
        # self.torch_device = get_torch_device()
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("STRING", {"default": "http://localhost:9379/v1/text2image"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
            "scheduler": (SCHEDULER_NAMES, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
            "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
            "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
            "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Diffusers"

    def sample(self, model, seed, steps, cfg, scheduler, positive, negative=None,
            width=512, height=512):

        print(model)
        print(positive)
        print(negative)

        req = {"prompt": positive,
            "num_images_per_prompt": 1,
        }

        try:
            res = requests.post(
                f"{model}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(req),
            )
            res.raise_for_status()
            res = res.json()
        except RequestException as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")

        image_str = res["images"][0]
        image_byte = base64.b64decode(image_str)
        image_io = BytesIO(image_byte)  # convert image to file-like object
        image = Image.open(image_io)   # img is now PIL Image object

        return (convert_images_to_tensors([image]),)


NODE_CLASS_MAPPINGS = {
    "DiffusersPipelineLoader": DiffusersPipelineLoader,
    "DiffusersVaeLoader": DiffusersVaeLoader,
    "DiffusersSchedulerLoader": DiffusersSchedulerLoader,
    "DiffusersModelMakeup": DiffusersModelMakeup,
    "DiffusersClipTextEncode": DiffusersClipTextEncode,
    "DiffusersSampler": DiffusersSampler,
    "DiffusersPipelineGaudi": DiffusersPipelineGaudi,
    "DiffusersModelPath": DiffusersModelPath,
    "DiffusersPipelineEndpointGaudi": DiffusersPipelineEndpointGaudi
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersPipelineLoader": "Diffusers Pipeline Loader",
    "DiffusersVaeLoader": "Diffusers Vae Loader",
    "DiffusersSchedulerLoader": "Diffusers Scheduler Loader",
    "DiffusersModelMakeup": "Diffusers Model Makeup",
    "DiffusersClipTextEncode": "Diffusers Clip Text Encode",
    "DiffusersSampler": "Diffusers Sampler",
    "DiffusersPipelineGaudi": "Diffusers Pipeline Gaudi",
    "DiffusersModelPath": "Diffusers Model Path",
    "DiffusersPipelineEndpointGaudi": "Diffusers Pipeline Endpoint Gaudi",
}
