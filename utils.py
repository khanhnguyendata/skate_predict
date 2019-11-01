import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib as mpl
import matplotlib.pyplot as plt


def train_test_split(scores, train_years, test_years):
    ''' Take all scores (for either male and female) 
    and split them into 4 different tables: 
    1. Season scores for training years
    2. World scores for training years
    3. Season scores for testing years
    4. World scores for testing years
    '''
    season_scores = scores.loc[scores['event']!='WR']
    world_scores = scores.loc[scores['event']=='WR']
    
    season_train = season_scores.loc[season_scores['year'].isin(train_years)]
    world_train = world_scores.loc[world_scores['year'].isin(train_years)]
    season_test = season_scores.loc[season_scores['year'].isin(test_years)]
    world_test = world_scores.loc[world_scores['year'].isin(test_years)]    
    
    return season_train, world_train, season_test, world_test


def get_yearly_scores(year, season_scores, world_scores):
    '''
    From season and world scores over multiple years,
    extract season scores of a given year (as pd.DataFrame with original index)
    and world scores of a given year (as pd.Series with skater name as index)
    '''
    yearly_season_scores = season_scores.loc[season_scores['year']==year].copy()
    yearly_world_scores = world_scores.loc[world_scores['year']==year, ['name', 'score']].set_index('name').squeeze()
    return yearly_season_scores, yearly_world_scores


def return_ranking(skater_scores, world_scores):
    '''
    Return predicted ranking and world rankings (as lists)
    from Series of corresponding scores.
    The reason why both rankings are returned at once is because
    not all skaters in the season will participate in the world championship,
    and vice versa. Therefore, only skaters who participated in both
    will be of interest.    
    '''
    skater_scores = skater_scores.sort_values(ascending=False)
    world_scores = world_scores.sort_values(ascending=False)
    skater_ranking = list(skater_scores.index.intersection(world_scores.index))
    world_ranking = list(world_scores.index.intersection(skater_scores.index))
    return skater_ranking, world_ranking


def calculate_kendall_tau(skater_ranking, world_ranking, verbose=True):
    '''
    Caluculate kendall's tau from two ranking lists of the same size
    '''
    # Generate set of ordered pairs from each ranking
    skater_pairs = set(combinations(skater_ranking, 2))
    world_pairs = set(combinations(world_ranking, 2))
    
    # Calculate number of total, concordant & discordant pairs and tau
    n_pairs = len(skater_pairs)
    n_concordant_pairs = len(skater_pairs & world_pairs)
    
    if verbose:
        print(f'There are {n_concordant_pairs} concordant_pairs out of {n_pairs} pairs')
        
    # Calculate Kendall's tau from pair counts
    tau = (2 * n_concordant_pairs - n_pairs) / n_pairs
    return tau 


def plot_multiple_rankings(fig, ax, rankings, labels, filepath=None, xfontsize=None, zorder=None):
    # Take multiple rankings as sorted list and plot them together (extension of plot_ranking)  
    cmap = mpl.cm.get_cmap('winter')

    last_ranking = rankings[-1]
    first_ranking = rankings[0]
    n_skaters = len(last_ranking)
    
    for index, skater in enumerate(last_ranking):
        # Get y-position (index), rank and color of each skater in last ranking in the list (usually worlds)
        last_rank = index + 1
        last_index = n_skaters - last_rank
        avg_color = cmap(last_rank/(n_skaters*1.1))
        
        # Get index and rank for this skater in first ranking (usually that of the current model)
        first_rank = first_ranking.index(skater) + 1
        first_index = n_skaters - first_rank
        
        # Get y-positions and rank for this skater for first to second-to-last rankings (to plot connecting lines)
        indicies = []        
        for ranking in rankings[:-1]: 
            subsequent_rank = ranking.index(skater) + 1
            subsequent_index = n_skaters - subsequent_rank
            indicies.append(subsequent_index)
            
        indicies.append(last_index)

        # Plot connecting lines between two rankings
        ax.plot(np.arange(len(rankings)), indicies,
                'o-', color=avg_color, zorder=zorder)

        # Plot text on both sides
        ax.text(-0.1, first_index, f'{skater} {first_rank}', ha='right', va='center', color=avg_color)
        ax.text(len(rankings)-0.9, last_index, f'{last_rank} {skater}', ha='left', va='center', color=avg_color)
        

    ax.set_xlim(-1, len(rankings))
    ax.set_xticks(np.arange(len(rankings)))
    ax.set_xticklabels(labels, fontsize=xfontsize)
    ax.set_yticks([])
    plt.tight_layout()
    
    if filepath:
        fig.savefig(filepath)


def calculate_rmse(predicted_scores, true_scores):
    squared_errors = (predicted_scores - true_scores)**2
    rmse = np.sqrt(squared_errors.mean())
    return rmse


class Model:
    def __init__(self):
        self.skater_scores = None
        self.model_ranking = None
        self.world_ranking = None
        self.predicted_season_scores = None

    def fit(self, season_scores):
        raise NotImplementedError('Must be overridden by specific fit method')

    def predict(self, season_scores):
        raise NotImplementedError('Must be overridden by specific predict method')
    
    def evaluate_rmse(self, season_scores):
        predicted_scores = self.predict(season_scores)
        squared_errors = (season_scores['score'].values - predicted_scores)**2
        rmse = np.sqrt(squared_errors.mean())
        return rmse
    
    def return_ranking(self, world_scores):
        skater_scores = self.skater_scores.sort_values(ascending=False)
        world_scores = world_scores.sort_values(ascending=False)
        skater_ranking = list(skater_scores.index.intersection(world_scores.index))
        world_ranking = list(world_scores.index.intersection(skater_scores.index))
        return skater_ranking, world_ranking
        
    def evaluate_kendall_tau(self, world_scores, verbose=True):
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
    
    def fit(self, season_scores):
        self.skater_scores = season_scores.groupby('name')['score'].mean()
        self.skater_scores.sort_values(ascending=False, inplace=True)

    def predict(self, season_scores):
        return self.skater_scores.loc[season_scores['name']].values


class NormedAverageScore(Model):
    def __init__(self):
        super().__init__()
        
    def fit(self, season_scores):
        season_scores = season_scores.copy()
        event_means = season_scores.groupby('event')['score'].mean().loc[season_scores['event']].values
        event_stds = season_scores.groupby('event')['score'].std().loc[season_scores['event']].values
        season_scores['score_normed'] = (season_scores['score'] - event_means) / event_stds
        
        self.skater_scores = season_scores.groupby('name')['score_normed'].mean()
        self.skater_scores.sort_values(ascending=False, inplace=True)   

    def predict(self, season_scores, event_stds, event_means):
        return (self.skater_scores.loc[season_scores['name']] * event_stds + event_means).values


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

    def predict(self, season_scores):
        broadcasted_skater_scores = self.skater_scores.loc[season_scores['name']].values
        broadcasted_event_scores = self.event_scores.loc[season_scores['event']].values
        return broadcasted_skater_scores + broadcasted_event_scores + self.baseline



class LogLinear(Linear):
    def __init__(self, lambda_reg=0):
        super().__init__(lambda_reg)
        
    def find_coefs(self, X, y):
        L = np.identity(n=len(X.T))
        L[0, 0] = 0
        coefs = np.linalg.inv(X.T @ X + self.lambda_reg * L) @ (X.T @ np.log(y))
        return coefs
    
    def predict(self, season_scores):
        broadcasted_skater_scores = self.skater_scores.loc[season_scores['name']].values
        broadcasted_event_scores = self.event_scores.loc[season_scores['event']].values
        return np.exp(broadcasted_skater_scores + broadcasted_event_scores + self.baseline)