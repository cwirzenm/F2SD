import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    data = pd.read_csv('big_data.csv')
    sns.set_theme()

    g = sns.FacetGrid(
            data.sort_values('score'),
            col='resolution',
            sharey=False,
            sharex=False,
            margin_titles=True,
            legend_out=False,
    )
    g.map(
            sns.barplot, 'name', 'score',
            order=['consistory avg', 'storydalle pororo', 'arldm pororo', 'temporalstory pororo',
                   'storydalle flintstones', 'arldm flintstones', 'temporalstory flintstones'],
            palette='tab10'
    )
    g.set(xticklabels=['CS', 'StDPo', 'ARLDMPo', 'TempPo', 'StDFS', 'ARLDMFS', 'TempFS'])
    plt.show()
