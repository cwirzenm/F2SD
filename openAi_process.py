from dbManager import DbManager
from openai import OpenAI
from PIL import Image as PILImage
from io import BytesIO
from PIL import Image
import requests
import datetime
import os


class TextToImagePipeline:

    def __init__(self, gpt_model="gpt-4", image_model="dall-e-3", image_size="1024x1024"):
        # init client
        self.client = OpenAI()

        # init database
        self.dbManager = DbManager()

        self.gpt_model = gpt_model
        self.image_model = image_model
        self.image_size = image_size

    def summarize_text(self, text):
        """
        Summarize the text using GPT-4.
        """
        response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                        {"role": "user", "content": f"Summarize the following text while keeping all the relevant details:\n\n{text}",}
                ],
                max_tokens=4000,
                temperature=0.5
        )
        summary = response.choices[0].message.content
        return summary

    def generate_image(self, prompt):
        """
        Generate an image using DALL-E based on a provided prompt.
        """
        response = self.client.images.generate(
                model=self.image_model,
                prompt=prompt,
                n=1,
                size=self.image_size
        )
        image_url = response.data[0].url
        return image_url

    @staticmethod
    def download_image(image_url):
        """
        Download the image from the provided URL.
        """
        response = requests.get(image_url)
        img = PILImage.open(BytesIO(response.content))
        return img

    @staticmethod
    def save_output(img: Image, parent_dir: str, page: int):
        if not os.path.exists(parent_dir): os.makedirs(parent_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"page_{page}_{timestamp}.jpg"
        img.save(os.path.join(parent_dir, filename))

    def get_story(self, book, page=None) -> list:
        query = {'book': book, 'page': page}
        return self.dbManager.get(query)

    def run_pipeline(self, text):
        """
        Process and visualize text using GPT-4 summarization and DALL-E image generation.
        """
        summary = self.summarize_text(text)
        print(f"Summary for image generation: {summary}")  # Preview the summary
        image_url = self.generate_image(summary)
        img = self.download_image(image_url)
        return img


if __name__ == "__main__":
    book = 'The Lord Of The Rings'
    pipeline = TextToImagePipeline()
    story = pipeline.get_story(book)
    for entry in story:
        print(f"Reading {book} {entry['page_number']}")
        print(f"{entry['page_text']}")
        img = pipeline.run_pipeline(entry['page_text'])

        # display the generated image
        # img.show(title="Generated Image")

        # save the image
        pipeline.save_output(img, f"output/fiction/{book}", entry['page_number'])
