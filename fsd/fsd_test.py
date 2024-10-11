from fsd_score import fsd_score
from dataset import CustomDataset
import functools


if __name__ == '__main__':
    # common params
    RES = (64, 64)
    custom_dataset = functools.partial(CustomDataset, res=RES)

    storydalle_gen_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\storydalle\\gen_pororo")
    storydalle_gt_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\storydalle\\gt_pororo")
    fsd_storydalle_pororo = fsd_score((storydalle_gt_pororo, storydalle_gen_pororo))
    print('StoryDALL-E FSD score on PororoSV:', fsd_storydalle_pororo)

    arldm_gen_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\arldm\\gen_pororo")
    arldm_gt_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\arldm\\gt_pororo")
    fsd_arldm_pororo = fsd_score((arldm_gt_pororo, arldm_gen_pororo))
    print('AR-LDM FSD score on PororoSV:', fsd_arldm_pororo)

    temporalstory_gen_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\temporalstory\\gen_pororo")
    temporalstory_gt_pororo = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\temporalstory\\gt_pororo")
    fsd_temporalstory_pororo = fsd_score((temporalstory_gt_pororo, temporalstory_gen_pororo))
    print('TemporalStory FSD score on PororoSV:', fsd_temporalstory_pororo)

    print()

    storydalle_gen_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\storydalle\\gen_flintstones")
    storydalle_gt_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\storydalle\\gt_flintstones")
    fsd_storydalle_flintstones = fsd_score((storydalle_gt_flintstones, storydalle_gen_flintstones))
    print('StoryDALL-E FSD score on FlintstonesSV:', fsd_storydalle_flintstones)

    arldm_gen_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\arldm\\gen_flintstones")
    arldm_gt_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\arldm\\gt_flintstones")
    fsd_arldm_flintstones = fsd_score((arldm_gt_flintstones, arldm_gen_flintstones))
    print('AR-LDM FSD score on FlintstonesSV:', fsd_arldm_flintstones)

    temporalstory_gen_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\temporalstory\\gen_flintstones")
    temporalstory_gt_flintstones = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\temporalstory\\gt_flintstones")
    fsd_temporalstory_flintstones = fsd_score((temporalstory_gt_flintstones, temporalstory_gen_flintstones))
    print('TemporalStory FSD score on FlintstonesSV:', fsd_temporalstory_flintstones)

    print()

    arldm_gen_vist = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\arldm\\gen_vist")
    arldm_gt_vist = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\fsd_data\\arldm\\gt_vist")
    fsd_arldm_vist = fsd_score((arldm_gt_vist, arldm_gen_vist))
    print('AR-LDM FSD score on VIST-SIS:', fsd_arldm_vist)

    noise_1 = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\noise\\gt_vist")
    noise_2 = custom_dataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\noise\\noise_1")
    fsd_noise = fsd_score((noise_1, noise_2))
    print('Noise FSD score:', fsd_noise)
