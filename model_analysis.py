import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 14})

try:
    df = pd.read_parquet('output/model_predictions_24-MAR-2025-19:13.pq')
except:
    df = pd.read_csv('output/model_predictions_24-MAR-2025-19:13.csv')

###
# Palette/style choices...
colors = {
    'Cumul. Point Winner': plt.cm.tab10(0),
    'Cumul. Game Winner': plt.cm.tab10(1),
    'Set Winner': plt.cm.tab10(2),
    'Cumul. Set Winner': plt.cm.tab10(3),
    'Cumul. Unf. Error': plt.cm.tab10(4),
    'Dynamic Model': '#000'
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

fg = sns.FacetGrid(data=df, col='Year')
fg.map_dataframe(sns.lineplot, x='Set', y='Accuracy', 
    style_order=order, size_order=order,
    style='Model', hue='Model', errorbar=None, marker='o', palette=colors
)


for axi in fg.axes.flatten():
    axi.set_title('')
    axi.axhline(0.5, c='#f00', ls='--', lw=0.5)
    axi.axhline(1, c='#333', ls='--', lw=0.5)
    axi.set(ylim=[0.2,1.05], xlim=[0.95,5.05], xticks=[1,2,3,4,5])

fg.set_titles('{col_name}', loc='left')

fg.add_legend()

fig = fg.figure
fig.text(0,1,'A', fontsize=24, weight='bold', ha='left', va='top')
fig.savefig('output/predictions_bymodel_byyear.pdf', bbox_inches='tight')
fig.show()

#################################################

# Question 2: If for a given set number, we *exclude* matches that were to 
# finish on that set, how would we perform?
mask = df['Set'] != df['total_sets']
df_sub = df[mask]

fg2 = sns.FacetGrid(data=df_sub, col='Year')
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

fg2.add_legend()

fig2 = fg2.figure
fig2.text(0,1,'B', fontsize=24, weight='bold', ha='left', va='top')
fig2.savefig('output/predictions_bymodel_byyear_NO_ENDERS.pdf', bbox_inches='tight')
fig2.show()


#####################################################

# Question 3: If we focus our attention on models that are 1 set away 
# from finishing, what are the predictive rates for each model?
df['Sets Remaining'] = -df['Sets Remaining']
dfs = df[df['Sets Remaining'] >= -4]

fg3 = sns.FacetGrid(data=df, col='Year')
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

fg3.add_legend()

fig3 = fg3.figure
fig3.text(0,1,'C', fontsize=24, weight='bold', ha='left', va='top')
fig3.savefig('output/predictions_bymodel_bysetsremaining.pdf', bbox_inches='tight')
fig3.show()

