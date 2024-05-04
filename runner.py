from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import torch
import os


class Dissertation:
    def __init__(self):
        # init variables
        self.lmd_pipe = None
        self.sdxl_refiner = None
        self.sdxl_negative_prompt = None

        # check if GPU is online
        self.torch_test()

        # init models
        self.init_lmdPlus()
        self.init_sdxl_refiner()

        # init database
        self.init_mongoDb()

    @staticmethod
    def torch_test():
        print("Torch version:", torch.__version__)
        assert torch.cuda.is_available()

    def init_lmdPlus(self, offload_model=True):
        self.lmd_pipe = DiffusionPipeline.from_pretrained(
                "longlian/lmd_plus",
                custom_pipeline="llm_grounded_diffusion",
                custom_revision="main",
                torch_dtype=torch.float16,
                variant="fp16"
        )
        if offload_model: self.lmd_pipe.enable_model_cpu_offload()
        else: self.lmd_pipe.to("cuda")

    def init_sdxl_refiner(self, offload_model=True):
        self.sdxl_negative_prompt = "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"
        self.sdxl_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
        )
        if offload_model: self.sdxl_refiner.enable_model_cpu_offload()
        else: self.sdxl_refiner.to("cuda")

    def init_mongoDb(self):
        pass

    def sdxl_refine(self, image, input_dict, refine_seed=None, refinement_step_ratio=0.5) -> Image:
        overall_prompt = input_dict['prompt']
        extra_neg_prompt = input_dict['neg_prompt']
        # g = torch.manual_seed(refine_seed)
        image = image.resize((1024, 1024), Image.LANCZOS)
        negative_prompt = extra_neg_prompt + ", " + self.sdxl_negative_prompt
        output = self.sdxl_refiner(overall_prompt, image=image, negative_prompt=negative_prompt, strength=refinement_step_ratio).images[0]
        # output = self.sdxl_refiner(overall_prompt, image=image, negative_prompt=negative_prompt, strength=refinement_step_ratio, generator=g).images[0]

        return output

    def save_output(self, img: Image, parent_dir: str, filename: str):
        # TODO make it so that it doesn't overwrite
        if not os.path.exists(parent_dir): os.makedirs(parent_dir)
        img.save(os.path.join(parent_dir, filename))
        print()

    def parse_input(self, prompt, response):
        phrases, boxes, bg_prompt, neg_prompt = self.lmd_pipe.parse_llm_response(response)
        return {
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "phrases": phrases,
            "boxes": boxes,
        }

    def run_pipeline(self):
        # todo read from db
        prompt = "a waterfall and a modern high speed train in a beautiful forest with fall foliage."
        response = """[('a waterfall', [100, 50, 200, 450]), ('a beautiful deer', [350, 250, 150, 200])] Background prompt: A dense forest surrounded by mountains Negative prompt:"""

        input_dict = self.parse_input(prompt, response)
        image = self.lmd_pipe(
                prompt=prompt,
                negative_prompt=input_dict['neg_prompt'],
                phrases=input_dict['phrases'],
                boxes=input_dict['boxes'],
                gligen_scheduled_sampling_beta=0.4,
                output_type="pil",
                num_inference_steps=10,
                lmd_guidance_kwargs={}
        ).images
        img = image[0]
        refined_img = self.sdxl_refine(img, input_dict)
        return refined_img

    def main(self):
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
                    output: Image = self.run_pipeline()
                    output.show()
                    self.save_output(output, f"output/{genre}/{title}", f"{page.split('.')[0]}.jpg")


project = Dissertation()
project.main()
