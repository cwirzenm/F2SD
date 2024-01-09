from diffusers import AutoPipelineForText2Image
import torch

print("Torch version:", torch.__version__)
print("Is CUDA enabled?", torch.cuda.is_available())

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A cinematic shot of a labrador in the forest."

image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

image.show()
