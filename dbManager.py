from pymongo import MongoClient


class DbManager:
    def __init__(self):
        client = MongoClient("mongodb://127.0.0.1:27017/")

        # dblist = self.client.list_database_names()
        # if "story-visualisation-database" in dblist:
        #     print('Database already exists')

        db = client["story-visualisation-database"]
        self.data = db['data']

    def insert(self, book, page_no, page_text, page_summary, prompt):
        mydict = {
                'book': book,
                'page_number': page_no,
                'page_text': page_text,
                'page_summary': page_summary,
                'prompt': prompt
        }
        x = self.data.insert_one(mydict)
        print(x)

    def getPages(self):
        query = {
                'book': 'The Great Gatsby'
        }
        row = self.data.find(query)
        print(row)
