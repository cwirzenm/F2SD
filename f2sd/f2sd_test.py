from f2sd_score import f2sd_score


if __name__ == '__main__':
    # consistory = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\consistory"
    # f2sd_consistory = f2sd_score(consistory)
    # print('ConsiStory F2SD score:', f2sd_consistory, end='\n\n')
    #
    # storydalle_flintstones = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\storydalle\\flintstones"
    # f2sd_storydalle_flintstones = f2sd_score(storydalle_flintstones)
    # print('StoryDALL-E F2SD score on FlintstonesSV:', f2sd_storydalle_flintstones, end='\n\n')
    #
    # arldm_flintstones = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\arldm\\flintstones"
    # f2sd_arldm_flintstones = f2sd_score(arldm_flintstones)
    # print('AR-LDM new F2SD score on FlintstonesSV:', f2sd_arldm_flintstones, end='\n\n')

    temporalstory_flintstones = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\temporalstory\\flintstones"
    f2sd_temporalstory_flintstones = f2sd_score(temporalstory_flintstones)
    print('TemporalStory new F2SD score on FlintstonesSV:', f2sd_temporalstory_flintstones, end='\n\n')

    # storydalle_pororo = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\storydalle\\pororo"
    # f2sd_storydalle_pororo = f2sd_score(storydalle_pororo)
    # print('StoryDALL-E F2SD score on PororoSV:', f2sd_storydalle_pororo, end='\n\n')
    #
    # arldm_pororo = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\arldm\\pororo"
    # f2sd_arldm_pororo = f2sd_score(arldm_pororo)
    # print('AR-LDM new F2SD score on PororoSV:', f2sd_arldm_pororo, end='\n\n')
    #
    # temporalstory_pororo = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\temporalstory\\pororo"
    # f2sd_temporalstory_pororo = f2sd_score(temporalstory_pororo)
    # print('TemporalStory new F2SD score on PororoSV:', f2sd_temporalstory_pororo, end='\n\n')

    noise = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\noise"
    f2sd_noise = f2sd_score(noise)
    print('Noisy F2SD:', f2sd_noise, end='\n\n')
