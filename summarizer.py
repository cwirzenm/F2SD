import replicate
import os


class Summarizer:
    def __init__(self):
        self.token = None
        self._api_test()

    def _api_test(self):
        self.token = os.environ.get('REPLICATE_API_TOKEN')
        assert self.token

    @staticmethod
    def get_prediction(prompt):
        input = {
                "prompt": prompt,
                "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                                   "You are a helpful summarizer"
                                   "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{"
                                   "prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        }

        prediction = "".join([event.data for event in replicate.stream(
                "meta/meta-llama-3-70b-instruct",
                input=input
        )])
        print(prediction)

        return prediction


s = Summarizer()
s.get_prediction(
        "About half way between West Egg and New York the motor road hastily joins the railroad and runs beside it for a quarter of a mile, so as to shrink away from a certain desolate area of land. This is a valley of ashes — a fantastic farm where ashes grow like wheat into ridges and hills and grotesque gardens; where ashes take the forms of houses and chimneys and rising smoke and, finally, with a transcendent effort, of men who move dimly and already crumbling through the powdery air. Occasionally a line of gray cars crawls along an invisible track, gives out a ghastly creak, and comes to rest, and immediately the ash-gray men swarm up with leaden spades and stir up an impenetrable cloud, which screens their obscure operations from your sight. But above the gray land and the spasms of bleak dust which drift endlessly over it, you perceive, after a moment, the eyes of Doctor T. J. Eckleburg. The eyes of Doctor T. J. Eckleburg are blue and gigantic — their irises are one yard high. They look out of no face, but, instead, from a pair of enormous yellow spectacles which pass over a nonexistent nose. Evidently some wild wag of an oculist set them there to fatten his practice in the borough of Queens, and then sank down himself into eternal blindness, or forgot them and moved away. But his eyes, dimmed a little by many paintless days, under sun and rain, brood on over the solemn dumping ground. The valley of ashes is bounded on one side by a small foul river, and, when the drawbridge is up to let barges through, the passengers on waiting trains can stare at the dismal scene for as long as half an hour. There is always a halt there of at least a minute, and it was because of this that I first met Tom Buchanan’s mistress."
)
