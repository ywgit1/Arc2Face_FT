from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)

from arc2face import CLIPTextModelWrapper, project_face_embs

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import pickle
import os

#filenames = ['chris.jpg', 'yan6.jpg', 'fangliang.jpg'] #["d3.jpg", "FR+EN.JPG"]
filenames = os.listdir('assets/examples2')
filenames = [img for img in filenames if img.lower().endswith('jpg') or 
        img.lower().endswith('png') or img.lower().endswith('bmp') or
        img.lower().endswith('jpeg') or img.lower().endswith('ppm')]
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
# id_embs = []
# for filename in filenames:
#     img = np.array(Image.open(f'assets/examples2/{filename}'))[:,:,::-1]
#     import os
#     fn = os.path.splitext(filename)[0]
#     faces = app.get(img)
#     faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
#     id_embs.append(faces["embedding"] / np.linalg.norm(faces["embedding"]))

# print(np.sum(id_embs[0] * id_embs[1]))

device = "cuda"
with open("external/emoca/cond_img.pkl", "rb") as file:
    cond_img = pickle.load(file)

# Arc2Face is built upon SD1.5
# The repo below can be used instead of the now deprecated 'runwayml/stable-diffusion-v1-5'
#base_model = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
base_model = 'models/stable_diffusion_v1_5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14'

encoder = CLIPTextModelWrapper.from_pretrained(
    'models', subfolder="encoder", torch_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    'models', subfolder="arc2face", torch_dtype=torch.float16
)

controlnet = ControlNetModel.from_pretrained(
    'models', subfolder="controlnet", torch_dtype=torch.float16
)

pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        #controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to('cuda')

#pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
#pipeline.disable_lora()
#pipeline.load_lora_weights("models/lora_weights/pytorch_lora_weights.safetensors")
pipeline.load_lora_weights("lora_models/checkpoint-100000/pytorch_lora_weights.safetensors")

for filename in filenames:
    img = np.array(Image.open(f'assets/examples2/{filename}'))[:,:,:3][:,:,::-1] # Load image as BGR
    import os
    fn = os.path.splitext(filename)[0]

    faces = app.get(img)
    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
    id_emb = torch.tensor(faces['embedding'], dtype=torch.float16)[None].cuda()
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
    id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder

    generator = torch.Generator(device=device).manual_seed(0)

    num_images = 4
    #images = pipeline(image=cond_img, prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images
    images = pipeline(prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images, \
                      cross_attention_kwargs={"scale": 1}).images

    for i, im in enumerate(images):
        im.save(os.path.join(output_folder, f"{fn}_{i}.png"))