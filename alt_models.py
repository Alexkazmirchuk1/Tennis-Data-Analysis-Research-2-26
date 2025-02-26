import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tennis_data
import dynamic_model1 as dm1

class MatchStats:
    def __init__(self, raw_data, match_to_examine):
        self.match = raw_data[raw_data['match_id'] == match_to_examine]
        
            
        self.player1_name = self.match['player1'].values[0]
        self.player2_name = self.match['player2'].values[0]
        self.player1_surname = self.match['p1_lastname'].values[0]
        self.player2_surname = self.match['p2_lastname'].values[0]
        
        self.names = [self.player1_name, self.player2_name]
        self.surnames = [self.player1_surname, self.player2_surname]
        
        # Identify points where sets change
        self.set_change_points = np.where(np.diff(self.match['set_no']) > 0)[0]
        self.set_change_points = np.append(self.set_change_points, self.match.shape[0]-1)
        
        # Match winner is the winner of final set
        self.match_winner = self.match['set_victor'].iloc[-1]
        
        # Extract various victor arrays
        self.set_victors = self.match['set_victor'][self.match['set_victor'] != 0]
        self.point_victors = self.match['point_victor'][self.match['point_victor'] != 0]
        self.game_victors = self.match['game_victor'].values
        
        # Properly track unforced errors (create separate arrays for each player)
        self.p1_unf_err = self.match['p1_unf_err'].values
        self.p2_unf_err = self.match['p2_unf_err'].values
        
        self.winner_id = self.match_winner
        self.winner_name = self.names[self.winner_id - 1]


class SetWinnerModel(MatchStats):
    '''
    At the end of set i, whoever won *that set* is predicted to be the winner
    of the overall match.
    '''
    def __init__(self, raw_data, match_to_examine):
        super().__init__(raw_data, match_to_examine)
    
    def fit(self):
        # No processing needed for this model
        pass
    
    def prediction(self):
        '''
        Output: prediction at the end of set 1,2,3,4,5 (if applicable) 
        to predict the winner of the match.
        '''
        # Get all set victors
        set_winners = self.set_victors.values
        
        # First set prediction is simply the winner of set 1
        # For subsequent sets, predict based on the previous set's winner
        predictions = np.full(len(set_winners), np.nan)
        for i in range(1, len(set_winners)):
            predictions[i] = set_winners[i-1]
            
        return predictions


class CumulativeSetWinnerModel(MatchStats):
    '''
    At the end of set i, whoever won *more sets until that point* is predicted
    to be the winner. 
    
    If it is a tie, whoever had previously been in the lead is marked as the 
    prediction.
    '''
    def __init__(self, raw_data, match_to_examine):
        super().__init__(raw_data, match_to_examine)
        self.p1_cumulative = None
        self.p2_cumulative = None
    
    def fit(self):
        self.p1_cumulative = np.cumsum(self.set_victors.values == 1)
        self.p2_cumulative = np.cumsum(self.set_victors.values == 2)
    
    def prediction(self):
        '''
        Output: prediction at the end of set 1,2,3,4,5 (if applicable) 
        to predict the winner of the match.
        '''
        # Initialize predictions array
        predictions = np.full(len(self.set_victors), np.nan)
        
        # First set has no previous information
        # For all other sets, make predictions based on cumulative set wins
        previous_leader = None
        
        for i in range(1, len(self.set_victors)):
            # Check who's leading in sets up to this point
            p1_sets = self.p1_cumulative[i-1]
            p2_sets = self.p2_cumulative[i-1]
            
            if p1_sets > p2_sets:
                predictions[i] = 1
                previous_leader = 1
            elif p2_sets > p1_sets:
                predictions[i] = 2
                previous_leader = 2
            else:
                # If tied, use previous leader
                predictions[i] = previous_leader
                
        return predictions


class CumulativePointWinnerModel(MatchStats):
    '''
    At the end of set i, whoever won *more points until that point* is predicted
    to be the winner.
    
    If it is a tie, whoever had previously been in the lead is marked as the
    prediction.
    '''
    def __init__(self, raw_data, match_to_examine):
        super().__init__(raw_data, match_to_examine)
        self.p1_point_cumulative = None
        self.p2_point_cumulative = None
    
    def fit(self):
        # Calculate cumulative points won by each player
        self.p1_point_cumulative = np.cumsum(self.point_victors.values == 1)
        self.p2_point_cumulative = np.cumsum(self.point_victors.values == 2)
    
    def prediction(self):
        '''
        Output: Prediction at the end of each set to predict the winner of the match.
        '''
        # Get set endpoints
        set_endpoints = self.set_change_points
        
        # Initialize predictions array
        predictions = np.full(len(self.set_victors), np.nan)
        previous_leader = None
        
        for i in range(1, len(self.set_victors)):
            # Find the point index at the end of previous set
            previous_set_end_idx = set_endpoints[i-1]
            
            # Check who's leading in points up to this point
            if previous_set_end_idx < len(self.p1_point_cumulative):
                p1_points = self.p1_point_cumulative[previous_set_end_idx]
                p2_points = self.p2_point_cumulative[previous_set_end_idx]
                
                if p1_points > p2_points:
                    predictions[i] = 1
                    previous_leader = 1
                elif p2_points > p1_points:
                    predictions[i] = 2
                    previous_leader = 2
                else:
                    # If tied, use previous leader if available
                    predictions[i] = previous_leader
        
        return predictions


class CumulativeGameWinnerModel(MatchStats):
    '''
    At the end of set i, whoever won *more games until that point* is predicted
    to be the winner.
    
    If it is a tie, whoever had previously been in the lead is marked as the
    prediction.
    '''
    def __init__(self, raw_data, match_to_examine):
        super().__init__(raw_data, match_to_examine)
        self.p1_game_cumulative = None
        self.p2_game_cumulative = None
    
    def fit(self):
        # Calculate cumulative games won by each player
        self.p1_game_cumulative = np.cumsum(self.game_victors == 1)
        self.p2_game_cumulative = np.cumsum(self.game_victors == 2)
    
    def prediction(self):
        '''
        Output: Prediction at the end of each set to predict the winner of the match.
        '''
        # Get set endpoints
        set_endpoints = self.set_change_points
        
        # Initialize predictions array
        predictions = np.full(len(self.set_victors), np.nan)
        previous_leader = None
        
        for i in range(1, len(self.set_victors)):
            # Find the game index at the end of previous set
            previous_set_end_idx = set_endpoints[i-1]
            
            # Check who's leading in games up to this point
            if previous_set_end_idx < len(self.p1_game_cumulative):
                p1_games = self.p1_game_cumulative[previous_set_end_idx]
                p2_games = self.p2_game_cumulative[previous_set_end_idx]
                
                if p1_games > p2_games:
                    predictions[i] = 1
                    previous_leader = 1
                elif p2_games > p1_games:
                    predictions[i] = 2
                    previous_leader = 2
                else:
                    # If tied, use previous leader if available
                    predictions[i] = previous_leader
        
        return predictions


class CumulativeUnfErrModel(MatchStats):
    '''
    At the end of set i, whoever has *fewer* unforced errors is predicted
    to be the winner.
    
    If it is a tie, whoever had previously been in the lead is marked as the
    prediction.
    '''
    def __init__(self, raw_data, match_to_examine):
        super().__init__(raw_data, match_to_examine)
        self.p1_error_cumulative = None
        self.p2_error_cumulative = None
    
    def fit(self):
        # Calculate cumulative errors by each player
        self.p1_error_cumulative = np.cumsum(self.p1_unf_err)
        self.p2_error_cumulative = np.cumsum(self.p2_unf_err)
    
    def prediction(self):
        '''
        Output: Prediction at the end of each set to predict the winner of the match.
        '''
        # Get set endpoints
        set_endpoints = self.set_change_points
        
        # Initialize predictions array
        predictions = np.full(len(self.set_victors), np.nan)
        previous_leader = None
        
        for i in range(1, len(self.set_victors)):
            # Find the point index at the end of previous set
            previous_set_end_idx = set_endpoints[i-1]
            
            # Check who has fewer errors up to this point
            if previous_set_end_idx < len(self.p1_error_cumulative):
                p1_errors = self.p1_error_cumulative[previous_set_end_idx]
                p2_errors = self.p2_error_cumulative[previous_set_end_idx]
                
                if p1_errors < p2_errors:
                    predictions[i] = 1
                    previous_leader = 1
                elif p2_errors < p1_errors:
                    predictions[i] = 2
                    previous_leader = 2
                else:
                    # If tied, use previous leader if available
                    predictions[i] = previous_leader if previous_leader is not None else 1
        
        return predictions

MAX_SETS = 4
def evaluate_models(df_raw, matches):
    # Store all models
    all_models = {
        "SetWinnerModel": SetWinnerModel,
        "CumulativeSetWinnerModel": CumulativeSetWinnerModel,
        "CumulativePointWinnerModel": CumulativePointWinnerModel,
        "CumulativeGameWinnerModel": CumulativeGameWinnerModel,
        "CumulativeUnfErrModel": CumulativeUnfErrModel,
        "DynamicTennisModel": dm1.DynamicTennisModel
    }
        
    # Counters
    count_correct = {model: np.zeros(MAX_SETS) for model in all_models}
    reach_count = {model: np.zeros(MAX_SETS) for model in all_models}
    all_results = np.zeros((MAX_SETS, len(all_models), len(matches)), dtype=float)
    
    # Iterate over each match
    total_matches = 0
    for k, match_id in enumerate(matches):
        try:
            # 2 or less sets
            stats = MatchStats(df_raw, match_id)
            if len(stats.set_victors) <= 2:
                continue
                
            total_matches += 1
            
            # Evaluate each model on this match
            for j, (model_name, model_class) in enumerate(all_models.items()):
                model_instance = model_class(df_raw, match_id)
                model_instance.fit()
                predictions = model_instance.prediction()
                
                # Store results
                num_sets = min(len(stats.set_victors), MAX_SETS)
                for i in range(num_sets):
                    if i+1 < len(predictions) and not np.isnan(predictions[i+1]):
                        is_correct = int(predictions[i+1] == stats.winner_id)
                        all_results[i, j, k] = is_correct
                        count_correct[model_name][i] += is_correct
                        reach_count[model_name][i] += 1
                    else:
                        all_results[i, j, k] = np.nan
                        
        except Exception as e:
            print(f"Error processing match {match_id}: {e}")
            continue
    
    # Calculate accuracy per set for each model
    accuracy = {}
    for model in all_models:
        with np.errstate(divide='ignore', invalid='ignore'):
            accuracy[model] = np.where(reach_count[model] > 0, 
                                    count_correct[model] / reach_count[model],
                                    np.nan)
    
    print(f"Successfully evaluated {total_matches} matches")
    
    # Display the accuracy
    for model, acc in accuracy.items():
        print(f"\nAccuracy for {model}:")
        for i, (set_acc, correct, possible) in enumerate(zip(acc, count_correct[model], reach_count[model])):
            if not np.isnan(set_acc):
                print(f"  Set {i+1}: {set_acc * 100:.1f}% ({int(correct)} correct out of {int(possible)})")
            else:
                print(f"  Set {i+1}: N/A")

    return accuracy, all_results


if __name__ == "__main__":
        
    df_raw = tennis_data.load_2022()
    matches = df_raw['match_id'].unique()
        
    # Print results for specific match
    my_match = matches[1]
    stats = MatchStats(df_raw, my_match)
        
    print("Actual winner:", stats.winner_id)
        
    print("\nModel predictions:")
    model_results = {}
        
    model1 = SetWinnerModel(df_raw, my_match)
    model1.fit()
    model_results["SetWinnerModel"] = model1.prediction()
        
    model2 = CumulativeSetWinnerModel(df_raw, my_match)
    model2.fit()
    model_results["CumulativeSetWinnerModel"] = model2.prediction()
        
    model3 = CumulativePointWinnerModel(df_raw, my_match)
    model3.fit()
    model_results["CumulativePointWinnerModel"] = model3.prediction()
        
    model4 = CumulativeGameWinnerModel(df_raw, my_match)
    model4.fit()
    model_results["CumulativeGameWinnerModel"] = model4.prediction()
        
    model5 = CumulativeUnfErrModel(df_raw, my_match)
    model5.fit()
    model_results["CumulativeUnfErrModel"] = model5.prediction()
        
    model6 = dm1.DynamicTennisModel(df_raw, my_match)
    model6.fit()
    model_results["DynamicTennisModel"] = model6.prediction()
        
    # Print predictions
    for model_name, predictions in model_results.items():
        print(f"{model_name}: {predictions}")
        
    # Evaluate all models
    print("\nEvaluating all models...")
    accuracy, all_results = evaluate_models(df_raw, matches)
        
    # Plot the results
    set_labels = [f"After Set {i+1}" for i in range(MAX_SETS)]
    plot_data = pd.DataFrame({
        'Model': [],
        'Set': [],
        'Percentage': []
    })
        
    for model, acc in accuracy.items():
        for i, set_acc in enumerate(acc):
            if not np.isnan(set_acc):
                plot_data = pd.concat([plot_data, pd.DataFrame({
                    'Model': [model],
                    'Set': [set_labels[i]],
                    'Percentage': [set_acc * 100]
                })], ignore_index=True)
        
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=plot_data, x='Set', y='Percentage', hue='Model', marker='o')
    plt.title('Model Prediction Accuracy by Set')
    plt.ylim(0, 100)
    plt.xlabel('Set')
    plt.ylabel('Percentage Correct')
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()
    