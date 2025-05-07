import dynamic_model1 as dm
import tennis_data
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np

from figstyle import *

# For the subplot image
include_subplot = True

# Load match data
df_raw = tennis_data.load_2021()
matches = df_raw['match_id'].unique()

# Matches for access
# 2021 = 124
# 2022 = 124
# 2023 = 126
# 2024 = 251
my_match = matches[124]

# Create and fit the model
model = dm.DynamicTennisModel(df_raw, my_match)
model.fit()

# Extract player names
p1name = model.match['player1'].iloc[0]
p2name = model.match['player2'].iloc[0]

if include_subplot:
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Plot performance momentum
    fig, ax[0] = model.graph_momentum(ax=ax[0])

    # Compute cumulative games won for each player
    point_victors = model.match['game_victor'].values
    p1_points = np.cumsum(point_victors == 1)
    p2_points = np.cumsum(point_victors == 2)

    color_p1 = 'tab:red'
    color_p2 = 'tab:blue'

    # Plot cumulative games (or points)
    ax[1].plot(p1_points, color=color_p1)
    ax[1].plot(p2_points, color=color_p2)

    # Fill between the lines
    ax[1].fill_between(np.arange(len(p1_points)), p1_points, p2_points,
                       where=p1_points >= p2_points, interpolate=True,
                       color=color_p1, alpha=0.3)
    ax[1].fill_between(np.arange(len(p1_points)), p1_points, p2_points,
                       where=p1_points < p2_points, interpolate=True,
                       color=color_p2, alpha=0.3)

    # Add decorations and labels
    ax[1].set(ylabel="Total Games Won")
    model.add_graph_decorations(ax[1])

    ax[0].set(ylabel="Performance", xlim=[0, model.match.shape[0]], ylim=[0, 1.1])
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    fig.align_ylabels(ax)
else:
    # Just momentum chart
    fig, ax = plt.subplots(figsize=(8, 4))
    fig, ax = model.graph_momentum(ax=ax)
    ax.set(ylabel="Performance", xlim=[0, model.match.shape[0]], ylim=[0, 1.1])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

fig.savefig('output/2021-champ.pdf', bbox_inches='tight')
fig.show()
