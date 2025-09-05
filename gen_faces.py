from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline
)

from arc2face import CLIPTextModelWrapper, project_face_embs
from time import sleep
import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import pickle
import os
from external.facexlib.utils.face_restoration_helper import FaceRestoreHelper
from arcface_onnx import ArcFaceONNX
from tqdm import tqdm

device = "cuda"
# filenames = ['chris_1.jpg', 'chris_3.jpg'] #["d3.jpg", "FR+EN.JPG"]
input_folder = 'E:/datasets/CASIA-WebFace-frontal-sampled'
filenames = os.listdir(input_folder)
filenames = [img for img in filenames if img.lower().endswith('jpg') or 
        img.lower().endswith('png') or img.lower().endswith('bmp') or
        img.lower().endswith('jpeg') or img.lower().endswith('ppm')]
output_folder = 'E:/datasets/CASIA-WebFace-arc2face-frontal'
os.makedirs(output_folder, exist_ok=True)

face_helper = FaceRestoreHelper(
        upscale_factor=2,
        face_size=256,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='jpg',
        use_parse=True,
        device=device,
        model_rootpath='D:/Projects/FaceSuperRes/GFPGAN/gfpgan/weights')

arcface_model = ArcFaceONNX('models/antelopev2/arcface.onnx')

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

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        text_encoder=encoder,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    )

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to('cuda')

#pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
#pipeline.disable_lora()
pipeline.load_lora_weights("models/lora_weights/pytorch_lora_weights.safetensors")
#pipeline.load_lora_weights("sd-model-finetuned-lora/checkpoint-49000/pytorch_lora_weights.safetensors")

    
for i, filename in enumerate(tqdm(filenames)):
    # img = np.array(Image.open(os.path.join(input_folder, f'{filename}')))[:,:,::-1] # Load image as BGR
    fn = os.path.splitext(filename)[0]
    subfolder = fn.split('_')[0]
    if os.path.exists(os.path.join(output_folder, subfolder)):
        continue

    face_helper.clean_all()
    face_helper.read_image(os.path.join(input_folder, f'{filename}'))
    face_helper.get_face_landmarks_5(only_keep_largest=True, eye_dist_threshold=5)
        
    # face_helper.cropped_faces.clear()
    face_helper.align_warp_face(used_template='arcface')
    if len(face_helper.cropped_faces) != 1:
        continue
    id_emb = arcface_model.get(face_helper.cropped_faces[0])
    id_emb = torch.tensor(id_emb, dtype=torch.float16)[None].cuda()
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
    id_emb = project_face_embs(pipeline, id_emb)    # pass through the encoder

    # generator = torch.Generator(device=device).manual_seed(0)

    num_images = 4
    # images = pipeline(image=cond_img, prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images).images
    images = pipeline(image=cond_img, prompt_embeds=id_emb, num_inference_steps=25, guidance_scale=3.0, num_images_per_prompt=num_images, \
                      cross_attention_kwargs={"scale": 0.5}).images
        
    os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)
    for i, im in enumerate(images):
        im.save(os.path.join(output_folder, subfolder, f"{fn}_{i}.png"))

    if i%50 == 100:
    	sleep(120)