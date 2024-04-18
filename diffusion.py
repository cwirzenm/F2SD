from diffusers import DiffusionPipeline
from PIL import Image
import torch
import os

print("Torch version:", torch.__version__)
print("Is CUDA enabled?", torch.cuda.is_available())
assert torch.cuda.is_available()


def get_diffusion_model() -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(
            "longlian/lmd_plus",
            custom_pipeline="llm_grounded_diffusion",
            custom_revision="main",
            torch_dtype=torch.float16,
            variant="fp16")
    pipe.to("cuda")
    return pipe


def run_pipeline(pipe: DiffusionPipeline, prompt) -> Image:
    prompt = "a waterfall and a modern high speed train in a beautiful forest with fall foliage."
    response = """
[('a waterfall', [100, 50, 200, 450]), ('a beautiful deer', [350, 250, 150, 200])] Background prompt: A dense forest surrounded by mountains Negative prompt:
    """
    phrases, boxes, bg_prompt, neg_prompt = pipe.parse_llm_response(response)
    # noinspection ALL
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        phrases=phrases,
        boxes=boxes,
        gligen_scheduled_sampling_beta=0.4,
        output_type="pil",
        num_inference_steps=25,
        lmd_guidance_kwargs={}
    ).images
    return image[0]


def save_output(img: Image, parent_dir: str, filename: str):
    # TODO make it so that it doesn't overwrite
    os.makedirs(parent_dir)
    img.save(os.path.join(parent_dir, filename))


def main():
    genres = os.listdir('tests')
    for genre in genres:
        titles = os.listdir(f'tests/{genre}')
        for title in titles:
            print(f'Reading {title}...')
            pages = os.listdir(f'tests/{genre}/{title}')
            if not pages: continue
            pages.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for page in pages:
                print(f'    Reading {page}...')
                with open(f'tests/{genre}/{title}/{page}', 'r', encoding="utf-8") as f: text = f.readlines()
                print(text)
                output: Image = run_pipeline(model, text)
                output.show()
                save_output(output, f"output/{genre}/{title}", f"{page.split('.')[0]}.jpg")

    return


model = get_diffusion_model()
main()

# Call the function to extract text from files in the current directory

# Print the extracted text

# pipe = DiffusionPipeline.from_pretrained(
#         "longlian/lmd_plus",
#         custom_pipeline="llm_grounded_diffusion",
#         custom_revision="main",
#         torch_dtype=torch.float16,
#         variant="fp16")
# pipe.to("cuda")

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=torch.device('cuda'))

# prompt = "a waterfall and a modern high speed train in a beautiful forest with fall foliage."
# response = """
# [('a waterfall', [100, 50, 200, 450]), ('a beautiful deer', [350, 250, 150, 200])] Background prompt: A dense forest surrounded by mountains Negative prompt:
# """


# ARTICLE = """
# An apple grew one day, during a winter frosty
# It grew on a tree in the lunar forest
#
# An apple ripened alone, under the lights northern
# It ripened on a tree among the eyes blindest
#
# An apple fell suddenly, by thoughts burdened
# It fell from a tree sharp sword-branches
#
# No one saw an apple, neath the snow blooded
# No one saw an appletree in this lunar forest
# """

# prompt = summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)
# print(prompt)

# phrases, boxes, bg_prompt, neg_prompt = pipe.parse_llm_response(response)
#
# # image = pipe(prompt=prompt[0]['summary_text'], num_inference_steps=4, guidance_scale=0.0).images[0]
# image = pipe(
#     prompt=prompt,
#     negative_prompt=neg_prompt,
#     phrases=phrases,
#     boxes=boxes,
#     gligen_scheduled_sampling_beta=0.4,
#     output_type="pil",
#     num_inference_steps=50,
#     lmd_guidance_kwargs={}
# ).images
#
# image[0].show()
#
# image = pipe(
#     prompt=prompt,
#     negative_prompt=neg_prompt,
#     phrases=phrases,
#     boxes=boxes,
#     gligen_scheduled_sampling_beta=0.4,
#     output_type="pil",
#     num_inference_steps=50,
#     lmd_guidance_kwargs={}
# ).images
#
# image[0].show()
