from fsd_score import fsd_score_without_gt
from dataset import CustomDataset
from yolov8x import YoloV8X
import pandas as pd
import functools


if __name__ == '__main__':
    # common params
    big_data = []

    # for RES in [(64, 64), (128, 128), (256, 256)]:
    for RES in [(256, 256)]:
        data = []

        fsd = functools.partial(fsd_score_without_gt, res=RES)

        consistory = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\new_fsd_data\\consistory"
        fsd_consistory = fsd(consistory)
        data.append({'name': 'consistory', 'score': fsd_consistory})
        print('ConsiStory FSD score:', fsd_consistory)

        print()

        storydalle_gen_pororo = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\new_fsd_data\\storydalle"
        fsd_storydalle_pororo = fsd(storydalle_gen_pororo)
        data.append({'name': 'storydalle', 'score': fsd_storydalle_pororo})
        print('StoryDALL-E new FSD score:', fsd_storydalle_pororo)

        arldm_gen_pororo = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\new_fsd_data\\arldm"
        fsd_arldm_pororo = fsd(arldm_gen_pororo)
        data.append({'name': 'arldm', 'score': fsd_arldm_pororo})
        print('AR-LDM new FSD score:', fsd_arldm_pororo)

        temporalstory_gen_pororo = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\new_fsd_data\\temporalstory"
        fsd_temporalstory_pororo = fsd(temporalstory_gen_pororo)
        data.append({'name': 'temporalstory', 'score': fsd_temporalstory_pororo})
        print('TemporalStory new FSD score:', fsd_temporalstory_pororo)

        print()
        for d in data:
            d['resolution'] = RES
            big_data.append(d)

    big_data = pd.DataFrame(big_data)
    big_data.to_csv('big_data.csv', index=False)
