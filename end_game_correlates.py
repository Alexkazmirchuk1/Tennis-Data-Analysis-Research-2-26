import numpy as np
import pandas
from matplotlib import pyplot as plt
from matplotlib import ticker
from sklearn import metrics
import tennis_data as td


from figstyle import *

# Load and prepare data
df_2021 = td.load_2021()
df_2022 = td.load_2022()
df_2023 = td.load_2023()
df_2024 = td.load_2024()

df = pandas.concat([df_2021, df_2022, df_2023, df_2024])
df = td.clean_data(df, min_sets_won=2)

def get_rocauc(y_true, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc

def clean_aucroc_plot(y_true, y_pred, predictor_str, myax=None):
    if myax is None:
        fig, myax = plt.subplots()
    
    fpr, tpr, auc = get_rocauc(y_true, y_pred)
    
    myax.plot(fpr, tpr, lw=2)
    myax.set(aspect='equal', xlabel='FPR', ylabel='TPR')
    myax.set_title(predictor_str + f' (AUC$={auc:.3})$', loc='left')
    myax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    myax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    myax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    myax.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    
    myax.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, p: str(np.round(x,2))))
    myax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, p: str(np.round(x,2))))
    myax.grid(True, which='minor', lw=0.5)
    myax.grid(True, which='major', lw=1)
    
    fill_color = plt.cm.tab10(0)
    fill_color = plt.cm.tab20(1)
    myax.fill_between(fpr, tpr, facecolor=fill_color, alpha=0.5)
    
    myax.set_aspect('equal')
    return myax

# Build a summary DataFrame
rows = []
for mid, df_tmp in df.groupby('match_id'):
    point_sum = df_tmp['point_victor'].value_counts()
    game_sum = df_tmp['game_victor'].value_counts()
    set_sum = df_tmp['set_victor'].value_counts()
    p1_unf_sum = df_tmp['p1_unf_err'].sum()
    p2_unf_sum = df_tmp['p2_unf_err'].sum()
    
    # New metrics
    p1_ace_sum = df_tmp['p1_ace'].sum()
    p2_ace_sum = df_tmp['p2_ace'].sum()
    p1_net_point = df_tmp['P1NetPointWon'].sum()
    p2_net_point = df_tmp['P2NetPointWon'].sum()
    p1_bp_won = df_tmp['P1BreakPointWon'].sum()
    p2_bp_won = df_tmp['P2BreakPointWon'].sum()
    p1_distance = df_tmp['P1DistanceRun'].sum()
    p2_distance = df_tmp['P2DistanceRun'].sum()
    p1_double_fault = df_tmp['P1DoubleFault'].sum()
    p2_double_fault = df_tmp['P2DoubleFault'].sum()
    
    p1n = df_tmp['player1'].iloc[0]
    p2n = df_tmp['player2'].iloc[0]
    rows.append([
        mid, p1n, p2n,
        point_sum.get(1, 0), point_sum.get(2, 0),
        game_sum.get(1, 0), game_sum.get(2, 0),
        set_sum.get(1, 0), set_sum.get(2, 0),
        p1_unf_sum, p2_unf_sum,
        p1_ace_sum, p2_ace_sum,
        p1_net_point, p2_net_point,
        p1_bp_won, p2_bp_won,
        p1_distance, p2_distance,
        p1_double_fault, p2_double_fault
    ])

df_summary = pandas.DataFrame(rows, 
    columns=[
        'match_id', 'player1', 'player2', 
        'points1', 'points2', 
        'games1', 'games2', 
        'sets1', 'sets2', 
        'unf_errs1', 'unf_errs2', 
        'aces1', 'aces2',
        'net_point1', 'net_point2',
        'bp_won1', 'bp_won2',
        'distance1', 'distance2',
        'double_fault1', 'double_fault2'
    ]
)

# Compute differential metrics and target variable
df_summary['Point Differential'] = df_summary['points2'] - df_summary['points1']
df_summary['Game Differential'] = df_summary['games2'] - df_summary['games1']
df_summary['Unforced. Error Differential'] = -1 * (df_summary['unf_errs2'] - df_summary['unf_errs1'])
df_summary['Ace Differential'] = df_summary['aces2'] - df_summary['aces1']
df_summary['Net Points Won Differential'] = df_summary['net_point2'] - df_summary['net_point1']
df_summary['Break Points Won Differential'] = df_summary['bp_won2'] - df_summary['bp_won1']
df_summary['Distance Ran Differential'] = df_summary['distance2'] - df_summary['distance1']
df_summary['Double Fault Differential'] = -1 * (df_summary['double_fault2'] - df_summary['double_fault1'])

df_summary['p2_wins'] = (df_summary['sets2'] > df_summary['sets1']).astype(int)

# Prepare the ROC AUC plots and horizontal bar chart in one row
fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
#fig.subplots_adjust(wspace=0.3)

# Plot the ROC AUC curves for 'Point differential' and 'Game differential'
for i, metric in enumerate(['Point Differential', 'Game Differential']):
    clean_aucroc_plot(df_summary['p2_wins'], df_summary[metric], metric, myax=axs[i])

# Compute AUC values for the remaining metrics and sort them
metrics_list = [
    'Unforced. Error Differential', 
    'Ace Differential',
    'Net Points Won Differential',
    'Break Points Won Differential',
    'Distance Ran Differential',
    'Double Fault Differential'
]
auc_values = []
for metric in metrics_list:
    _, _, auc = get_rocauc(df_summary['p2_wins'], df_summary[metric])
    auc_values.append(auc)

#
metrics_list_abbrv = [' '.join(li.split()[:-1]+['Diff.']) for li in metrics_list]

# Combine, sort (highest first), then reverse the order for horizontal bar chart 
auc_data = list(zip(metrics_list_abbrv, auc_values))
auc_data.sort(key=lambda x: x[1], reverse=True)
sorted_metrics, sorted_auc = zip(*auc_data)
sorted_metrics = list(sorted_metrics)[::-1]  # reverse for horizontal bar chart (highest at top)
sorted_auc = list(sorted_auc)[::-1]

# Plot the horizontal bar chart of ROC AUC values
bars = axs[2].barh(sorted_metrics, sorted_auc, color=plt.cm.tab20(1))

# Annotate the bars with the AUC values
for bar in bars:
    width = bar.get_width()
    if False:
        axs[2].annotate(f'{width:.3f}', 
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),  # small horizontal offset
                        textcoords="offset points",
                        ha='left', va='center', fontsize=10, fontweight='bold')

axs[2].set(xlabel='AUC value', xlim=[0,1])
axs[2].set_title('AUC values for other metrics', loc='left')
#axs[2].set_xlabel('AUC Value', fontsize=12)
#axs[2].set_title('ROC AUC Values for Differential Metrics', fontsize=10, fontweight='bold')
#axs[2].set_xlim(0, 1)
axs[2].grid(axis='x', linestyle='--', zorder=-100)
#axs[2].set_aspect('equal')

# why do i need to force this...
a0b = axs[0].get_position().bounds
a2b = axs[2].get_position().bounds
#axs[2].set_position([a2b[0], a0b[1], a2b[2], a0b[3]])

fig.savefig('output/combined_plots.pdf', bbox_inches='tight')

fig.show()
