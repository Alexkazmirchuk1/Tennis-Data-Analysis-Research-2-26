import numpy as np
import pandas

import tennis_data as td

df_2021 = td.load_2021()
df_2022 = td.load_2022()
df_2023 = td.load_2023()
df_2024 = td.load_2024()

# join data
df = pandas.concat([df_2021, df_2022, df_2023, df_2024])

def clean_aucroc_plot(y_true, y_pred, predictor_str, myax=None):
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    
    if ax is None:
        fig,myax = plt.subplots()
        
    fpr,tpr,_ = metrics.roc_curve(y_true, y_pred)
    auc = np.trapz(tpr,fpr)
    
    myax.plot(fpr, tpr, lw=2)
    myax.set(aspect='equal', xlabel='FPR', ylabel='TPR')
    #myax.text(0.5, 0.5, f"AUC={auc:.3}", bbox={'ec':'#333', 'fc':'#ccc'})
    myax.set_title(predictor_str + f' (AUC$={auc:.3})$', loc='left')
    myax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    myax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    myax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    myax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    
    myax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,p: str(np.round(x,2))))
    myax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x,p: str(np.round(x,2))))
    myax.grid(True, which='minor', lw=0.5)
    myax.grid(True, which='major', lw=1)
    
    fill_color = plt.cm.tab10(0)
    fill_color = np.array(fill_color)**0.25 # lighten
    myax.fill_between(fpr, tpr, facecolor=fill_color, alpha=0.5)
    
    myax.set_aspect('equal')
    return myax
    
#######

rows = []
for mid,df_tmp in df.groupby('match_id'):
    #
    point_sum = df_tmp['point_victor'].value_counts()
    game_sum = df_tmp['game_victor'].value_counts()
    set_sum = df_tmp['set_victor'].value_counts()
    match_sum = int( set_sum.get(2,0) > set_sum.get(1,0) )
    p1_unf_sum = df_tmp['p1_unf_err'].sum()
    p2_unf_sum = df_tmp['p2_unf_err'].sum()
    
    p1n = df_tmp['player1'].iloc[0]
    p2n = df_tmp['player2'].iloc[0]
    rows.append( [mid, p1n, p2n, point_sum.get(1,0), point_sum.get(2,0), game_sum.get(1,0), game_sum.get(2,0), set_sum.get(1,0), set_sum.get(2,0), p1_unf_sum, p2_unf_sum] )
#

df_summary = pandas.DataFrame(rows, 
    columns=['match_id', 'player1', 'player2', 'points1', 'points2', 'games1', 'games2', 'sets1', 'sets2', 'unf_errs1', 'unf_errs2']
)

#
# comparative
df_summary['Point differential'] = (df_summary['points2'] - df_summary['points1'])
df_summary['Game differential'] = (df_summary['games2'] - df_summary['games1'])

df_summary['Unf. error differential'] = -1*(df_summary['unf_errs2'] - df_summary['unf_errs1'])

df_summary['p2_wins'] = (df_summary['sets2'] > df_summary['sets1']).astype(int)

#
if __name__=="__main__":
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    #import seaborn
    from sklearn import metrics
    
    fig,ax = plt.subplots(1,3, figsize=(12,4), constrained_layout=True)
    
    for i,l in enumerate(['Unf. error differential', 'Point differential', 'Game differential']):
        clean_aucroc_plot(df_summary['p2_wins'], df_summary[l], l, myax=ax[i])
    
    fig.show()
    
