import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 14})

#########
SAVE_FIGURES = True
FIG_HEIGHT_IN = 4
FIG_WIDTH_IN = 4

##########

try:
    df = pd.read_parquet('output/model_predictions_31-MAR-2025-11:20.pq')
except:
    df = pd.read_csv('output/model_predictions_31-MAR-2025-11:20.csv')

###
# Palette/style choices...
colors = {
    'Cumul. Point Winner': plt.cm.tab10(0),
    'Cumul. Game Winner': plt.cm.tab10(1),
    'Set Winner': plt.cm.tab10(2),
    'Cumul. Set Winner': plt.cm.tab10(3),
    'Cumul. Unf. Error': plt.cm.tab10(4),
    'Dynamic Model': '#000',
    #
    'M': '#909',
    'W': '#0b0'
}


order=[
    'Dynamic Model',
    'Set Winner',
    'Cumul. Set Winner',
    'Cumul. Game Winner',
    'Cumul. Point Winner',
    'Cumul. Unf. Error'
]

#####

# 
# Question 1: across all matches, regardless of total number of matches, 
# how does each model perform?

#fig,ax = plt.subplots()
df['Prediction'] = df['match_victor'] == df['model_prediction']
df['ours'] = 1+(df['model_name'] == 'Dynamic Model')
#sns.lineplot(data=df, x='set_no', y='Prediction', hue='model_name', errorbar=None, ax=ax)

df.rename(
    columns={'Prediction': 'Accuracy', 
        'year': 'Year', 
        'model_name': 'Model', 
        'set_no': 'Set',
        'sets_until_end': 'Sets Remaining'}, 
        inplace=True)

fg = sns.FacetGrid(data=df, col='Year', height=FIG_HEIGHT_IN, aspect=FIG_WIDTH_IN/FIG_HEIGHT_IN)
fg.map_dataframe(sns.lineplot, x='Set', y='Accuracy', 
    style_order=order, size_order=order,
    style='Model', hue='Model', errorbar=None, marker='o', palette=colors
)


for axi in fg.axes.flatten():
    axi.set_title('')
    axi.axhline(0.5, c='#f00', ls='--', lw=0.5)
    axi.axhline(1, c='#333', ls='--', lw=0.5)
    axi.set(ylim=[0.2,1.05], xlim=[0.95,5.05], xticks=[1,2,3,4,5])

fg.set_titles('{col_name}', loc='left', zorder=100)

fg.figure.text(0,1,'A', fontsize=24, weight='bold', ha='left', va='top')

#################################################

# Question 2: If for a given set number, we *exclude* matches that were to 
# finish on that set, how would we perform?
mask = df['Set'] != df['total_sets']
df_sub = df[mask]

fg2 = sns.FacetGrid(data=df_sub, col='Year', height=FIG_HEIGHT_IN, aspect=FIG_WIDTH_IN/FIG_HEIGHT_IN)
fg2.map_dataframe(sns.lineplot, x='Set', y='Accuracy', 
    style_order=order, size_order=order,
    style='Model', hue='Model', errorbar=None, marker='o', palette=colors
)


for axi in fg2.axes.flatten():
    axi.set_title('')
    axi.axhline(0.5, c='#f00', ls='--', lw=0.5)
    axi.axhline(1, c='#333', ls='--', lw=0.5)
    axi.set(ylim=[0.2,1.05], xlim=[0.95, 5.05], xticks=[1,2,3,4,5])

fg2.set_titles('{col_name}', loc='left')

fg2.figure.text(0,1,'B', fontsize=24, weight='bold', ha='left', va='top')

#####################################################

# Question 3: If we focus our attention on models that are 1 set away 
# from finishing, what are the predictive rates for each model?
df['Sets Remaining'] = -df['Sets Remaining']
dfs = df[df['Sets Remaining'] >= -4]

fg3 = sns.FacetGrid(data=dfs, col='Year', height=FIG_HEIGHT_IN, aspect=FIG_WIDTH_IN/FIG_HEIGHT_IN)
fg3.map_dataframe(sns.lineplot, x='Sets Remaining', y='Accuracy', 
    style_order=order, size_order=order,
    style='Model', hue='Model', errorbar=None, marker='o', palette=colors
)


for axi in fg3.axes.flatten():
    axi.set_title('')
    axi.axhline(0.5, c='#f00', ls='--', lw=0.5)
    axi.axhline(1, c='#333', ls='--', lw=0.5)
    axi.set(ylim=[0.2,1.05], xlim=[-4.05,0.05], xticks=[-4,-3,-2,-1,0])

fg3.set_titles('{col_name}', loc='left')

fg3.figure.text(0,1,'C', fontsize=24, weight='bold', ha='left', va='top')

#####################################################

# Question 4: Do predictions on matches in the women's division differ 
# significantly from those in the open division, from the perspective 
# of "games until finish"?

#df['Sets Remaining'] = -df['Sets Remaining']
dfs = df[df['Model'] == 'Dynamic Model']
dfs = dfs[dfs['Sets Remaining'] >= -2] # games that finish in at most 3 sets
#dfs1 = df[df['program']=='W']
#dfs2 = df[df['program']=='M']



fg4 = sns.FacetGrid(data=dfs, col='Year', height=FIG_HEIGHT_IN, aspect=FIG_WIDTH_IN/FIG_HEIGHT_IN)
fg4.map_dataframe(sns.lineplot, x='Sets Remaining', y='Accuracy', 
    style='program', hue='program', errorbar=None, marker='o', palette=colors
)


for axi in fg4.axes.flatten():
    axi.set_title('')
    axi.axhline(0.5, c='#f00', ls='--', lw=0.5)
    axi.axhline(1, c='#333', ls='--', lw=0.5)
    axi.set(ylim=[0.2,1.05], xlim=[-4.05,0.05], xticks=[-4,-3,-2,-1,0])

fg4.set_titles('{col_name}', loc='left')

fg4.figure.text(0,1,'D', fontsize=24, weight='bold', ha='left', va='top')

if True:
    # force the figures to have the same layout as the first.

    for fgi in [fg,fg2,fg3,fg4]:
        fgi.add_legend(loc='upper right', frameon=True, edgecolor='k')
        fgi.figure.subplots_adjust(right=0.89)
        
        #fgi.figure.show()
#

if SAVE_FIGURES:
    fg.figure.savefig('output/predictions_bymodel_byyear.pdf', bbox_inches='tight')
    fg2.figure.savefig('output/predictions_bymodel_byyear_NO_ENDERS.pdf', bbox_inches='tight')
    fg3.figure.savefig('output/predictions_bymodel_bysetsremaining.pdf', bbox_inches='tight')
    fg4.figure.savefig('output/predictions_dynamic_byprogram.pdf', bbox_inches='tight')



