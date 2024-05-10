from pymongo import MongoClient


class DbManager:
    def __init__(self):
        client = MongoClient("mongodb://127.0.0.1:27017/")
        db = client["story-visualisation-database"]
        self.data = db['data']

    def get(self, query):
        return self.data.find_one(query)

    def emit(self, book, page_no, page_text, page_summary, prompt):
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
