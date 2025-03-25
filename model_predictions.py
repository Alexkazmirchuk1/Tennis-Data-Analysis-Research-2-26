import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

import tennis_data
import dynamic_model1 as dm1
import alt_models

# Runs predictions for, but does not analyze the results of, the suite of models, 
# across all data.
#
# CSV file generated is in a "long" format with columns
# year | match_id | match_victor | total_sets | set_no | model_name | model_prediction

# TODO: move to tennis_data as another loader function?
_year = []
_to_concat = []
for yr in [2021, 2022, 2023, 2024]:
    _method = f'load_{yr}'
    _df_raw = getattr(tennis_data, _method)()
    _to_concat.append( tennis_data.clean_data(_df_raw) )
    
    _year.append( np.repeat(yr, _to_concat[-1].shape[0]) )

_year = np.concatenate(_year)
df = pd.concat(_to_concat)
df['year'] = _year

# To iterate over...

models = [
    alt_models.CumulativePointWinnerModel,
    alt_models.CumulativeGameWinnerModel,
    alt_models.SetWinnerModel,
    alt_models.CumulativeSetWinnerModel,
    alt_models.CumulativeUnfErrModel,
    dm1.DynamicTennisModel
]

matches = df['match_id'].unique()

#

columns =  ["year" , "match_id", "match_victor" , "total_sets" , "set_no", "model_name", "model_prediction"]
#dtypes = ['str', 'str', 'str', 'int', 'int', 'str', 'str']
rows = []
# Main loop
for match in matches:
    _df = df[df['match_id'] == match]
    yr = df['year'].iloc[0]
    match_victor = _df['set_victor'].iloc[-1]
    final_set_no = _df['set_no'].iloc[-1]
    
    for _m in models:
        model = _m(_df, match) # instantiate a copy
        model.fit()
        pred = model.prediction()
        for i,p in enumerate(pred):
            if np.isnan(p):
                break
            rows.append([
                yr,
                match,
                match_victor,
                final_set_no,
                i+1,
                model.short_name,
                str(int(p))
            ])

df_predictions = pd.DataFrame(rows, columns=columns, dtype=str)
df_predictions['total_sets'] = df_predictions['total_sets'].astype(int)
df_predictions['set_no'] = df_predictions['set_no'].astype(int)

#
df_predictions['matches_until_end'] = df_predictions['total_sets'] - df_predictions['set_no']

if True:
    import datetime
    dt = datetime.datetime.now()
    tstamp = dt.strftime('%d-%b-%Y_%H:%M').upper()
    
    df_predictions.to_csv(f'output/model_predictions_{tstamp}.csv', index=None)
    df_predictions.to_parquet(f'output/model_predictions_{tstamp}.pq', index=None)
    
