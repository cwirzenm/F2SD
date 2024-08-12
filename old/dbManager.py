from pymongo import MongoClient
import os


class DbManager:
    def __init__(self):
        client = MongoClient("mongodb://127.0.0.1:27017/")
        db = client["story-visualisation-database"]
        self.data = db['data']

    def get(self, query):
        return [x for x in self.data.find(query)]

    def emit(self, book, page_no, page_text, page_summary='', prompt=''):
        query = {
                'book': book,
                'page_number': page_no
        }
        if self.get(query):
            update = {
                    '$set': {
                            'page_text': page_text,
                            'page_summary': page_summary,
                            'prompt': prompt
                    }
            }
            x = self.data.update_one(query, update)
        else:
            insert = {
                    'book': book,
                    'page_number': page_no,
                    'page_text': page_text,
                    'page_summary': page_summary,
                    'prompt': prompt
            }
            x = self.data.insert_one(insert)

        print(x)

    def populate_from_dir(self, genre, title):
        print(f'Reading {title}...')
        pages = os.listdir(f'tests/{genre}/{title}')
        # pages.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for page in pages:
            print(f'    Reading {page}...')
            with open(f'tests/{genre}/{title}/{page}', 'r', encoding="utf-8") as f: text = "".join(f.readlines())
            self.emit(title, page.removesuffix('.txt'), text)


if __name__ == "__main__":
    dbManager = DbManager()

    genre = 'fiction'
    title = 'Tristan and Iseult'
    dbManager.populate_from_dir(genre, title)

    genre = 'drama'
    title = 'Hamlet'
    dbManager.populate_from_dir(genre, title)
