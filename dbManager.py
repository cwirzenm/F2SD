from pymongo import MongoClient


class DbManager:
    def __init__(self):
        self.client = MongoClient("mongodb://127.0.0.1:27017/")

        dblist = self.client.list_database_names()
        if "story-visualisation-database" in dblist:
            print('Database already exists')
        self.db = self.client["story-visualisation-database"]

        # self.test()
        self.read()

    def test(self):
        col = self.db['samples']

        mydict = {
                'book': 'The Great Gatsby',
                'page number': '20',
                'text': 'About half way between West Egg and New York the motor road hastily joins the railroad and runs beside it for a quarter of a mile, so as to shrink away from a certain desolate area of land. This is a valley of ashes — a fantastic farm where ashes grow like wheat into ridges and hills and grotesque gardens; where ashes take the forms of houses and chimneys and rising smoke and, finally, with a transcendent effort, of men who move dimly and already crumbling through the powdery air. Occasionally a line of gray cars crawls along an invisible track, gives out a ghastly creak, and comes to rest, and immediately the ash-gray men swarm up with leaden spades and stir up an impenetrable cloud, which screens their obscure operations from your sight. But above the gray land and the spasms of bleak dust which drift endlessly over it, you perceive, after a moment, the eyes of Doctor T. J. Eckleburg. The eyes of Doctor T. J. Eckleburg are blue and gigantic — their irises are one yard high. They look out of no face, but, instead, from a pair of enormous yellow spectacles which pass over a nonexistent nose. Evidently some wild wag of an oculist set them there to fatten his practice in the borough of Queens, and then sank down himself into eternal blindness, or forgot them and moved away. But his eyes, dimmed a little by many paintless days, under sun and rain, brood on over the solemn dumping ground. The valley of ashes is bounded on one side by a small foul river, and, when the drawbridge is up to let barges through, the passengers on waiting trains can stare at the dismal scene for as long as half an hour. There is always a halt there of at least a minute, and it was because of this that I first met Tom Buchanan’s mistress.',
                'prompt': ''
        }

        x = col.insert_one(mydict)
        print(x)

    def read(self):
        col = self.db['samples']
        query = {
                'book': 'The Great Gatsby'
        }
        row = col.find(query)
        print(row)


DbManager()
