import numpy as np
import pandas as pd
from itertools import combinations


def get_yearly_scores(year, season_scores, world_scores):
    '''
    From season and world scores over multiple years,
    extract season scores of a given year (as pd.DataFrame with original index)
    and world scores of a given year (as pd.Series with skater name as index)
    '''
    yearly_season_scores = season_scores.loc[season_scores['year']==year].copy()
    yearly_world_scores = world_scores.loc[world_scores['year']==year, ['name', 'score']].set_index('name').squeeze()
    return yearly_season_scores, yearly_world_scores


class Model:
    def __init__(self):
        self.skater_scores = None
        self.model_ranking = None
        self.world_ranking = None
        self.predicted_season_scores = None
    
    def evaluate_rmse(self, season_scores):
        squared_errors = (season_scores['score'].values - self.predicted_season_scores)**2
        rmse = np.sqrt(squared_errors.mean())
        return rmse
    
    def return_ranking(self, world_scores):
        skater_scores = self.skater_scores.sort_values(ascending=False)
        world_scores = world_scores.sort_values(ascending=False)
        skater_ranking = list(skater_scores.index.intersection(world_scores.index))
        world_ranking = list(world_scores.index.intersection(skater_scores.index))
        return skater_ranking, world_ranking
        
    def evaluate_kendall_tau(self, world_scores, verbose=True):         
        skater_scores = self.skater_scores.squeeze()
        self.model_ranking, self.world_ranking = self.return_ranking(world_scores)
        
        skater_pairs = set(combinations(self.model_ranking, 2))
        world_pairs = set(combinations(self.world_ranking, 2))
        n_pairs = len(skater_pairs)
        n_concordant_pairs = len(set(skater_pairs) & set(world_pairs))
        if verbose:
            print(f'There are {n_concordant_pairs} concordant_pairs out of {n_pairs} pairs')
        tau = (2 * n_concordant_pairs - n_pairs) / n_pairs
        return tau, n_concordant_pairs, n_pairs
    
    def evaluate_over_years(self, years, season_df, world_df, **kwargs):
        taus = []
        rmses = []
        concordant_pairs = []
        n_pairs = []
        for year in years:
            season_scores, world_scores = get_yearly_scores(year, season_df, world_df)
            self.fit(season_scores, **kwargs)
            rmse = self.evaluate_rmse(season_scores)
            tau, concordant_pair, n_pair = self.evaluate_kendall_tau(world_scores, verbose=False)
            
            rmses.append(rmse)
            taus.append(tau)
            concordant_pairs.append(concordant_pair)
            n_pairs.append(n_pair)
        return pd.DataFrame({'year': years, 'rmse': rmses, 
                             'tau': taus, 'conc': concordant_pairs, 'pairs': n_pairs}).sort_values(by='year')


class AverageScore(Model):
    def __init__(self):
        super().__init__()
        
    def predict_season_scores(self, season_scores):
        self.predicted_season_scores = self.skater_scores.loc[season_scores['name']].values
    
    def fit(self, season_scores):
        self.skater_scores = season_scores.groupby('name')['score'].mean()
        self.skater_scores.sort_values(ascending=False, inplace=True)
        self.predict_season_scores(season_scores)        


class NormedAverageScore(Model):
    def __init__(self):
        super().__init__()
    
    def predict_season_scores(self, season_scores, event_stds, event_means):
        self.predicted_season_scores = (self.skater_scores.loc[season_scores['name']] * event_stds + event_means).values
        
    def fit(self, season_scores):
        season_scores = season_scores.copy()
        event_means = season_scores.groupby('event')['score'].mean().loc[season_scores['event']].values
        event_stds = season_scores.groupby('event')['score'].std().loc[season_scores['event']].values
        season_scores['score_normed'] = (season_scores['score'] - event_means) / event_stds
        
        self.skater_scores = season_scores.groupby('name')['score_normed'].mean()
        self.skater_scores.sort_values(ascending=False, inplace=True)
        
        self.predict_season_scores(season_scores, event_stds, event_means)        


class Linear(Model):
    def __init__(self, lambda_reg=0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.event_scores = None
        self.baseline = None
        
    def find_coefs(self, X, y):
        L = np.identity(n=X.shape[1])
        L[0, 0] = 0
        coefs = np.linalg.inv(X.T @ X + self.lambda_reg * L) @ (X.T @ y)
        return coefs
    
    def predict_season_scores(self, season_scores):
        broadcasted_skater_scores = self.skater_scores.loc[season_scores['name']].values
        broadcasted_event_scores = self.event_scores.loc[season_scores['event']].values
        self.predicted_season_scores = broadcasted_skater_scores + broadcasted_event_scores + self.baseline
        
    def fit(self, season_scores):
        dummies = pd.get_dummies(season_scores[['name', 'event']], prefix=['', ''], prefix_sep='', drop_first=True)
        unique_skaters = season_scores['name'].unique()
        unique_events = season_scores['event'].unique()
        
        dummies_skater_count = len(unique_skaters) - 1
        dummies_skaters = dummies.columns[:dummies_skater_count]
        dummies_events = dummies.columns[dummies_skater_count:]

        dropped_skater = list(set(unique_skaters) - set(dummies_skaters))[0]
        dropped_event = list(set(unique_events) - set(dummies_events))[0]

        X = dummies.values
        X = np.insert(X, 0, 1, axis=1)
        y = season_scores['score'].values
        coefs = self.find_coefs(X, y)

        self.baseline = coefs[0]    
        self.skater_scores = pd.Series(coefs[1:dummies_skater_count+1], index=dummies_skaters)
        self.event_scores = pd.Series(coefs[dummies_skater_count+1:], index=dummies_events)
        self.skater_scores[dropped_skater] = 0
        self.event_scores[dropped_event] = 0
        
        self.skater_scores.sort_values(ascending=False, inplace=True)
        self.event_scores.sort_values(ascending=False, inplace=True)        
        
        self.predict_season_scores(season_scores)


class LogLinear(Linear):
    def __init__(self, lambda_reg=0):
        super().__init__(lambda_reg)
        
    def find_coefs(self, X, y):
        L = np.identity(n=len(X.T))
        L[0, 0] = 0
        coefs = np.linalg.inv(X.T @ X + self.lambda_reg * L) @ (X.T @ np.log(y))
        return coefs
    
    def predict_season_scores(self, season_scores):
        broadcasted_skater_scores = self.skater_scores.loc[season_scores['name']].values
        broadcasted_event_scores = self.event_scores.loc[season_scores['event']].values
        self.log_predicted_season_scores = broadcasted_skater_scores + broadcasted_event_scores + self.baseline
        self.predicted_season_scores = np.exp(self.log_predicted_season_scores)