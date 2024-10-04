from fsd_score import fsd_score, fsd_score_without_gt
from dataset import CustomDataset
from background_sub import BackGroundSubtraction
from yolov8x import YoloV8X
import pandas as pd
import functools


if __name__ == '__main__':
    # common params
    # RES = (64, 64)

    big_data = []

    for RES in [(64, 64), (128, 128), (256, 256)]:
        data = []

        # consistory = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\testing_consistory"
        # fsd_consistory = fsd_score_without_gt(consistory, res=RES)
        # data.append({'name': 'consistory avg', 'score': fsd_consistory})
        # print('ConsiStory avg FSD score:', fsd_consistory)

        print()
        custom_dataset = functools.partial(CustomDataset, res=RES)

        # storydalle_gen_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\storydalle\\gen_pororo")
        # storydalle_gt_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\storydalle\\gt_pororo")
        # fsd_storydalle_pororo = fsd_score((storydalle_gt_pororo, storydalle_gen_pororo))
        # data.append({'name': 'storydalle pororo', 'score': fsd_storydalle_pororo})
        # print('StoryDALL-E FSD score on PororoSV:', fsd_storydalle_pororo)
        #
        # arldm_gen_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\arldm\\gen_pororo")
        # arldm_gt_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\arldm\\gt_pororo")
        # fsd_arldm_pororo = fsd_score((arldm_gt_pororo, arldm_gen_pororo))
        # data.append({'name': 'arldm pororo', 'score': fsd_arldm_pororo})
        # print('AR-LDM FSD score on PororoSV:', fsd_arldm_pororo)

        temporalstory_gen_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\temporalstory\\gen_pororo")
        temporalstory_gt_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\temporalstory\\gt_pororo")
        fsd_temporalstory_pororo = fsd_score((temporalstory_gt_pororo, temporalstory_gen_pororo))
        data.append({'name': 'temporalstory pororo', 'score': fsd_temporalstory_pororo})
        print('TemporalStory FSD score on PororoSV:', fsd_temporalstory_pororo)

        print()

        storydalle_gen_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\storydalle\\gen_flintstones")
        storydalle_gt_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\storydalle\\gt_flintstones")
        fsd_storydalle_flintstones = fsd_score((storydalle_gt_flintstones, storydalle_gen_flintstones))
        data.append({'name': 'storydalle flintstones', 'score': fsd_storydalle_flintstones})
        print('StoryDALL-E FSD score on FlintstonesSV:', fsd_storydalle_flintstones)

        arldm_gen_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\arldm\\gen_flintstones")
        arldm_gt_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\arldm\\gt_flintstones")
        fsd_arldm_flintstones = fsd_score((arldm_gt_flintstones, arldm_gen_flintstones))
        data.append({'name': 'arldm flintstones', 'score': fsd_arldm_flintstones})
        print('AR-LDM FSD score on FlintstonesSV:', fsd_arldm_flintstones)

        temporalstory_gen_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\temporalstory\\gen_flintstones")
        temporalstory_gt_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\temporalstory\\gt_flintstones")
        fsd_temporalstory_flintstones = fsd_score((temporalstory_gt_flintstones, temporalstory_gen_flintstones))
        data.append({'name': 'temporalstory flintstones', 'score': fsd_temporalstory_flintstones})
        print('TemporalStory FSD score on FlintstonesSV:', fsd_temporalstory_flintstones)

        print()
        for d in data:
            d['resolution'] = RES
            big_data.append(d)

    big_data = pd.DataFrame(big_data)
    big_data.to_csv('big_data.csv', index=False)

    # arldm_gen_vist = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\arldm\\gen_vist")
    # arldm_gt_vist = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\arldm\\gt_vist")
    # fsd_arldm_vist = fsd_score((arldm_gt_vist, arldm_gen_vist))
    # print('AR-LDM FSD score on VIST-SIS:', fsd_arldm_vist)
