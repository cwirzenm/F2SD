LAYOUT_PROMPT_TEMPLATE = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"

You are an intelligent bounding box generator. I will provide you with a summary of a scene. Your task is to generate the bounding boxes for the objects 
mentioned in the caption, along with a background prompt describing the scene. The images are of size 512x512. The top-left corner has coordinate [0, 0]. 
The bottom-right corner has coordinate [512, 512]. The bounding boxes should not overlap or go beyond the image boundaries. Each bounding box should be 
in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and should not include more than one object. 
Do not put objects that are already provided in the bounding boxes into the background prompt. Do not include non-existing or excluded objects in the 
background prompt. Use "A realistic scene" as the background prompt if no background is given in the prompt. If needed, you can make reasonable guesses. 
Don't use quotes (') or double quotes (") inside the strings.
Please refer to the example below for the desired format.

Caption: A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky
Objects: [('a green car', [21, 281, 211, 159]), ('a blue truck', [269, 283, 209, 160]), ('a red air balloon', [66, 8, 145, 135]), ('a bird', [296, 42, 143, 100])]
Background prompt: A realistic landscape scene
Negative prompt: 

Caption: A realistic top-down view of a wooden table with two apples on it
Objects: [('a wooden table', [20, 148, 472, 216]), ('an apple', [150, 226, 100, 100]), ('an apple', [280, 226, 100, 100])]
Background prompt: A realistic top-down view
Negative prompt: 

Caption: A realistic scene of three skiers standing in a line on the snow near a palm tree
Objects: [('a skier', [5, 152, 139, 168]), ('a skier', [278, 192, 121, 158]), ('a skier', [148, 173, 124, 155]), ('a palm tree', [404, 105, 103, 251])]
Background prompt: A realistic outdoor scene with snow
Negative prompt: 

Caption: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
Objects: [('a steam boat', [232, 225, 257, 149]), ('a jumping pink dolphin', [21, 249, 189, 123])]
Background prompt: An oil painting of the sea
Negative prompt: 

Caption: A cute cat and an angry dog without birds
Objects: [('a cute cat', [51, 67, 271, 324]), ('an angry dog', [302, 119, 211, 228])]
Background prompt: A realistic scene
Negative prompt: birds

Caption: Two pandas in a forest without flowers
Objects: [('a panda', [30, 171, 212, 226]), ('a panda', [264, 173, 222, 221])]
Background prompt: A forest
Negative prompt: flowers

Caption: An oil painting of a living room scene without chairs with a painting mounted on the wall, a cabinet below the painting, and two flower vases on the cabinet
Objects: [('a painting', [88, 85, 335, 203]), ('a cabinet', [57, 308, 404, 201]), ('a flower vase', [166, 222, 92, 108]), ('a flower vase', [328, 222, 92, 108])]
Background prompt: An oil painting of a living room scene
Negative prompt: chairs

Caption: {prompt}
Objects: 

<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
"""

SUMMARY_PROMPT_TEMPLATE = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n

You are an intelligent descriptor. I will provide you with a page from a work of literature. Your task is to describe what is happening on that page.

<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
"""