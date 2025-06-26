import numpy as np
import pandas as pd
import tennis_data

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

table = []
for program,dfp in df.groupby('program'):
    for year,dfpy in dfp.groupby('year'):
        for match,dfpym in dfpy.groupby('match_id'):
            set_score = [dfpym['set_victor'].value_counts().get(i,0) for i in [1,2]]
            winner = 1+np.argmax(set_score)
            num_sets = sum(set_score)
            
            #row=['program','year','match_id','num_sets']
            table.append( [program,year,match,num_sets] )

table = pd.DataFrame(table, columns=['program', 'year', 'match_id', 'num_sets'])
table['ones'] = 1
#table.pivot_table(aggfunc='count')
#values, index, columns, aggfunc
set_counts = table.pivot_table(index='num_sets', columns='year', values='ones', aggfunc='sum')

print(set_counts)
