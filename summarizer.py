import replicate
import os
from dbManager import DbManager
from prompt_template import LAYOUT_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE


class Summarizer:
    def __init__(self):
        self.token = None
        self._api_test()
        self.dbManager = DbManager()

    def _api_test(self):
        self.token = os.environ.get('REPLICATE_API_TOKEN')
        assert self.token

    @staticmethod
    def generate(input):
        model_output = "".join([event.data for event in replicate.stream(
                "meta/meta-llama-3-70b-instruct",
                input=input
        )])
        print(model_output)
        return model_output

    def generate_summary(self, source):
        input = {
                "prompt": source,
                "prompt_template": SUMMARY_PROMPT_TEMPLATE
        }
        return self.generate(input)

    def generate_layout(self, summary):
        input = {
                "prompt": summary,
                "prompt_template": LAYOUT_PROMPT_TEMPLATE
        }
        return self.generate(input)

    def populate_db(self):
        genre = 'fiction'
        title = 'The Great Gatsby'

        # genres = os.listdir('tests')
        # for genre in genres:
        # titles = os.listdir(f'tests/{genre}')
        # for title in titles:

        print(f'Reading {title}...')
        pages = os.listdir(f'tests/{genre}/{title}')
        # if not pages: continue
        pages.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for page in pages:
            print(f'    Reading {page}...')
            with open(f'tests/{genre}/{title}/{page}', 'r', encoding="utf-8") as f: text = "".join(f.readlines())
            summary = self.generate_summary(text)
            prompt = self.generate_layout(summary)
            self.dbManager.emit(title, int(page.split('_')[-1].split('.')[0]), text, summary, prompt)


s = Summarizer()
s.populate_db()
