from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from dbManager import DbManager
from PIL import Image
import torch
import datetime
import os


class LmdPlusSDXL:
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
        self.dbManager = DbManager()

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

    def sdxl_refine(self, image, input_dict, refine_seed=None, refinement_step_ratio=0.5) -> Image:
        overall_prompt = input_dict['prompt']
        extra_neg_prompt = input_dict['neg_prompt']
        # g = torch.manual_seed(refine_seed)
        image = image.resize((1024, 1024), Image.LANCZOS)
        negative_prompt = extra_neg_prompt + ", " + self.sdxl_negative_prompt
        output = self.sdxl_refiner(overall_prompt, image=image, negative_prompt=negative_prompt, strength=refinement_step_ratio).images[0]
        # output = self.sdxl_refiner(overall_prompt, image=image, negative_prompt=negative_prompt, strength=refinement_step_ratio, generator=g).images[0]

        return output

    @staticmethod
    def save_output(img: Image, parent_dir: str, page: int):
        # TODO make it so that it doesn't overwrite
        if not os.path.exists(parent_dir): os.makedirs(parent_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"page_{page}_{timestamp}.jpg"
        img.save(os.path.join(parent_dir, filename))

    def parse_input(self, prompt, response: str):
        phrases, boxes, bg_prompt, neg_prompt = self.lmd_pipe.parse_llm_response(response)
        return {
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "phrases": phrases,
            "boxes": boxes,
        }

    def get_test_data(self, query):
        return self.dbManager.get(query)

    def run_pipeline(self, title):
        prompt = "a waterfall and a modern high speed train in a beautiful forest with fall foliage."
        response = """[('a waterfall', [100, 50, 200, 450]), ('a beautiful deer', [350, 250, 150, 200])] Background prompt: A dense forest surrounded by mountains Negative prompt:"""

        data = self.get_test_data({'book': title, 'page_number': 133})

        for test in data:
            # test2 = {
            #         'page_summary': prompt,
            #         'prompt': response
            # }

            input_dict = self.parse_input(test['page_summary'], test['prompt'])
            img = self.lmd_pipe(
                    prompt=test['prompt'],
                    negative_prompt=input_dict['neg_prompt'],
                    phrases=input_dict['phrases'],
                    boxes=input_dict['boxes'],
                    gligen_scheduled_sampling_beta=0.4,
                    output_type="pil",
                    num_inference_steps=25,
                    lmd_guidance_kwargs={}
            ).images[0]
            refined_img = self.sdxl_refine(img, input_dict)

            refined_img.show()
            self.save_output(refined_img, f"output/fiction/{title}", test['page_number'])


project = LmdPlusSDXL()
project.run_pipeline('The Great Gatsby')
