
import pandas as pd

# TODO: only if necessary, make this operating system agnostic.
DATA_FOLDER = 'data/'

# TODO: programmatically filter to this
five_sets_2021 = [
    '2021-wimbledon-1104', 
    '2021-wimbledon-1105', 
    '2021-wimbledon-1107', 
    '2021-wimbledon-1108', 
    '2021-wimbledon-1122', 
    '2021-wimbledon-1123', 
    '2021-wimbledon-1125', 
    '2021-wimbledon-1126', 
    '2021-wimbledon-1127', 
    '2021-wimbledon-1136', 
    '2021-wimbledon-1140', 
    '2021-wimbledon-1142', 
    '2021-wimbledon-1144', 
    '2021-wimbledon-1158', 
    '2021-wimbledon-1163', 
    '2021-wimbledon-1214',
    '2021-wimbledon-1215', 
    '2021-wimbledon-1216', 
    '2021-wimbledon-1223', 
    '2021-wimbledon-1316', 
    '2021-wimbledon-1402', 
    '2021-wimbledon-1403', 
    '2021-wimbledon-1406', 
    '2021-wimbledon-1408', 
    '2021-wimbledon-1502'
]

def load_2021(FILE=DATA_FOLDER+'2021-wimbledon-data.csv', exclusions=['2021-wimbledon-1149']):
    '''
    Outputs a Pandas dataframe, loading the file while excluding 
    the given match_ids.
    
    Inputs: FILE: string; location of 2021 Wimbledon data.
            exclusions: list of strings; match_ids to ignore. 
            Default includes match_id 2021-wimbledon-1149
            
    TODO: write code to ease pulling matches with 5 full sets (currently 
    hard-coded in Main.ipynb as of 29 Jul 2024)
    '''
    df = pd.read_csv(FILE)
    mask = ~ df['match_id'].isin(exclusions)
    df = df[mask]
    
    # helps searching by last name/family names
    # magic string processing syntax.
    df['p1_lastname'] = df['player1'].str.split().str[1:].str.join(' ')
    df['p2_lastname'] = df['player2'].str.split().str[1:].str.join(' ')
    
    return df
    
def load_2022(FILE=DATA_FOLDER+'2022-wimbledon-data.csv', exclusions=[]):
    '''
    Outputs a Pandas dataframe, loading the file while excluding 
    the given match_ids.
    
    Inputs: FILE: string; location of 2022 Wimbledon data.
            exclusions: list of strings; match_ids to ignore. 
            Default is an empty list (include all 2022 data)
            
    TODO: write code to ease pulling matches with 5 full sets (currently 
    hard-coded in Main.ipynb as of 29 Jul 2024)
    '''
    df = pd.read_csv(FILE)
    mask = ~ df['match_id'].isin(exclusions)
    df = df[mask]
    
    # helps searching by last name/family names
    # magic string processing syntax.
    df['p1_lastname'] = df['player1'].str.split().str[1:].str.join(' ')
    df['p2_lastname'] = df['player2'].str.split().str[1:].str.join(' ')
    
    return df


def load_2023(FILE=DATA_FOLDER+'2023-wimbledon-data.csv', exclusions=[]):
    '''
    Outputs a Pandas dataframe, loading the file while excluding 
    the given match_ids.
    
    Inputs: FILE: string; location of 2023 Wimbledon data.
            exclusions: list of strings; match_ids to ignore. 
            Default is an empty list (include all 2023 data)
            
    TODO: write code to ease pulling matches with 5 full sets (currently 
    hard-coded in Main.ipynb as of 29 Jul 2024)
    '''
    df = pd.read_csv(FILE)
    mask = ~ df['match_id'].isin(exclusions)
    df = df[mask]
    
    # helps searching by last name/family names
    # magic string processing syntax.
    df['p1_lastname'] = df['player1'].str.split().str[1:].str.join(' ')
    df['p2_lastname'] = df['player2'].str.split().str[1:].str.join(' ')
    
    return df

def load_2024(FILE=DATA_FOLDER+'2024-wimbledon-data.csv', exclusions=[]):
    '''
    Outputs a Pandas dataframe, loading the file while excluding 
    the given match_ids.
    
    Inputs: FILE: string; location of 2024 Wimbledon data.
            exclusions: list of strings; match_ids to ignore. 
            Default is an empty list (include all 2024 data)
            
    TODO: write code to ease pulling matches with 5 full sets (currently 
    hard-coded in Main.ipynb as of 29 Jul 2024)
    '''
    df = pd.read_csv(FILE)
    mask = ~ df['match_id'].isin(exclusions)
    df = df[mask]
    
    # helps searching by last name/family names
    # magic string processing syntax.
    df['p1_lastname'] = df['player1'].str.split().str[1:].str.join(' ')
    df['p2_lastname'] = df['player2'].str.split().str[1:].str.join(' ')
    
    return df

def clean_data(mydf, min_sets_won=2, max_sets=5, verbose=False):
    '''
    Given a dataset presumed to come from the tennis data sources, this does 
    the following:
        
        0. Adds a column titled "program" for the program - M (men's) or W (women's)
            inferred from the original match_id (digit after "wimbledon-" being 1 or 2 
            respectively); or 'NA' if other/missing.
        1. Fills the final row of 'set_victor' with the corresponding 'point_winner'
            under the assumption that those games completed and the final point of a 
            match is always going to the victor.
        2. Excludes any match with fewer than min_sets_won (default: 3)
        3. Excludes any match with greater than max_sets (default: 5).
    '''
    rebuild = []
    mapping = {'1': 'M', '2': 'W'}
    programs = []
    for m,df_sub in mydf.groupby('match_id'):
        # get the men's/women's program
        try:
            match_id_num = m.split('-')[-1]
            program = mapping.get(match_id_num[0], 'NA')
        except:
            program = 'NA'
        
        # fill final value
        if df_sub['set_victor'].iloc[-1] ==0:
            w = df_sub['point_victor'].iloc[-1]
            df_sub.loc[df_sub.index[-1], 'set_victor'] = w
            if verbose:
                print(f"match_id {m}: final set_victor 0 imputed with {w}.")
        # skip if insufficient data (presumed cancelled game)
        set_result = df_sub['set_victor'].value_counts()
        if set_result.get(1,0)<min_sets_won and set_result.get(2,0)<min_sets_won:
            if verbose:
                print(f"Skipping match_id {m} for insufficient sets won by either player.")
            continue
        if set_result.get(1,0) + set_result.get(2,0) > max_sets:
            if verbose:
                print(f"Skipping match_id {m} for going past {max_sets} sets.")
            continue
        programs.append([program]*df_sub.shape[0])
        rebuild.append(df_sub)
    
    _df = pd.concat(rebuild)
    _df['program'] = sum(programs,[]) # concatenate list of lists
    return _df
    
