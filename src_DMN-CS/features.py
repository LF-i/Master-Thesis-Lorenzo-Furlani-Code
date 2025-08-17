import abc
import pandas as pd
import numpy as np

# Code is adapted from https://github.com/kieranjwood/trading-momentum-transformer/tree/master

class AbstractFeatures(abc.ABC):
    
    @abc.abstractmethod
    def calc_features(self, close_srs: pd.DataFrame, open_srs=None, 
                      high_srs=None, low_srs=None, volume_srs=None):
        pass


class DeepMomentumFeatures(AbstractFeatures):
    
    def __init__(self, vol_lookback: int=60, vol_target: float=0.15):
        self._vol_lookback  = vol_lookback # span of 60 days (Lim, Zohren & Roberts, 2020, p.3)
        self._vol_target    = vol_target   # annualized volatility target (Tan, Roberts & Zohren, 2023, p.3)

    # Calculate return over the past day_offset days
    def _calc_returns(self, srs: pd.Series, day_offset: int=1, log_transform: bool=False) -> pd.Series:
        if not log_transform:
            returns = srs / srs.shift(day_offset) - 1.0
        else:
            returns = np.log(srs) - np.log(srs.shift(day_offset))
        return returns

    # Calculate ex-ante volatility estimate - computed using an exponentially weighted 
    # moving standard deviation with a 60-day span (Lim, Zohren & Roberts, 2020, p.3)
    def _calc_daily_vol(self, daily_returns: pd.Series) -> pd.Series:
        s = daily_returns.ewm(span=self._vol_lookback, min_periods=self._vol_lookback).std()
        return s

    # Calculate volatility scaled returns
    def _calc_vol_scaled_returns(self, daily_returns: pd.Series, daily_vol: pd.Series) -> pd.Series:
        annualized_vol = daily_vol*np.sqrt(252)
        return daily_returns * self._vol_target / annualized_vol.shift(1) # returns of today x ex-ante vola of yesterday

    # Calculate normalized returns
    def _calc_normalised_returns(self, srs: pd.Series, daily_vol: pd.Series, day_offset: int) -> pd.Series:
        return (
            self._calc_returns(srs, day_offset) # returns with day_offset scale
            / daily_vol
            / np.sqrt(day_offset) # scale daily_vol to make it correspond to day_offset scale
        )

    # NB: only the values corresponding to the key close_srs are used
    def calc_features(self, close_srs: pd.Series, open_srs=None, 
                      high_srs=None, low_srs=None, volume_srs=None) -> pd.DataFrame:

        df_asset = pd.DataFrame() # create pd.DataFrame where to collect all results

        df_asset["close"]           = close_srs # collect close prices
        df_asset["srs"]             = df_asset["close"]
        df_asset["daily_returns"]   = self._calc_returns(df_asset["srs"]) # collect daily returns
        df_asset["daily_vol"]       = self._calc_daily_vol(df_asset["daily_returns"]) # collect daily ex-ante volatility estimate

        # Vol scaling and shift to be next day's returns
        df_asset["target_returns"] = self._calc_vol_scaled_returns(
            df_asset["daily_returns"], df_asset["daily_vol"]
            ).shift(-1)
        df_asset["target_returns_nonscaled"] = df_asset["daily_returns"].shift(-1)
        # NB: the target returns are already shifted one way forward !!!
        
        df_asset["norm_daily_return"]       = self._calc_normalised_returns(df_asset["srs"], df_asset["daily_vol"], 1)
        df_asset["norm_weekly_return"]      = self._calc_normalised_returns(df_asset["srs"], df_asset["daily_vol"], 5)
        df_asset["norm_biweekly_return"]    = self._calc_normalised_returns(df_asset["srs"], df_asset["daily_vol"], 10)
        df_asset["norm_monthly_return"]     = self._calc_normalised_returns(df_asset["srs"], df_asset["daily_vol"], 21)
        df_asset["norm_quarterly_return"]   = self._calc_normalised_returns(df_asset["srs"], df_asset["daily_vol"], 63)
        df_asset["norm_biannual_return"]    = self._calc_normalised_returns(df_asset["srs"], df_asset["daily_vol"], 126)
        df_asset["norm_annual_return"]      = self._calc_normalised_returns(df_asset["srs"], df_asset["daily_vol"], 252)
    
        return df_asset


class MACDFeatures(AbstractFeatures):

    def __init__(self, trend_combinations=[(8,24), (16,48), (32,96)]):
        self._trend_combinations = trend_combinations

    def _calc_signal(self, srs: pd.Series, short_timescale: int, long_timescale: int) -> pd.Series:
        """Calculate MACD signal for a signal short/long timescale combination
        Args:
            srs ([type]): series of prices
            short_timescale ([type]): short timescale
            long_timescale ([type]): long timescale
        Returns:
            float: MACD signal
        """

        def _calc_halflife(timescale: int) -> float:
            return np.log(0.5) / (np.log(1 - 1 / timescale) + 1e-9) # add 1e-9 buffer to make sure it is not divided by zero 

        macd = (
            srs.ewm(halflife=_calc_halflife(short_timescale)).mean()
            - srs.ewm(halflife=_calc_halflife(long_timescale)).mean()
        )
        q = macd / (srs.rolling(63).std(ddof=1).bfill() + 1e-9)   # 63-day rolling standard deviation
        return q / (q.rolling(252).std(ddof=1).bfill() + 1e-9)    # 252-day rolling standard deviation

    # NB: only the values corresponding to the key close_srs are used
    def calc_features(self, close_srs: pd.Series, open_srs=None, high_srs=None, low_srs=None, volume_srs=None) -> pd.DataFrame:
        srs = close_srs
        feats = pd.DataFrame()

        for comb in self._trend_combinations:
            f = self._calc_signal(srs, comb[0], comb[1])
            feats['macd_{}_{}'.format(comb[0], comb[1])] = f

        return feats


class DatetimeFeatures(AbstractFeatures):
    def __init__(self, **kwargs):
        pass

    # Returns the day of week, day of month, and month of the Date index
    def calc_features(self, close_srs: pd.Series, open_srs=None, high_srs=None, low_srs=None, volume_srs=None) -> pd.DataFrame:
        
        df_asset                    = pd.DataFrame(index=close_srs.index)
        df_asset["day_of_week"]     = close_srs.index.dayofweek
        df_asset["week_of_year"]    = close_srs.index.isocalendar().week
        df_asset["year"]            = close_srs.index.year
        df_asset["month_of_year"]   = close_srs.index.month
        df_asset["date"]            = close_srs.index
        
        return df_asset


class DefaultFeatureCreator:
    def __init__(self, prices_df: pd.DataFrame, symbols: list[str], features: list[abc.ABCMeta], params: list[dict], 
                 half_winsore: int=252, vol_threshold: int=5, cut_nan_samples: int=253):
        
        self._prices_df = prices_df
        if len(features) == 0:
            raise Exception('Number of features cannot be zero!')
        self._features = features
        self._params = params
        self._half_winsore = half_winsore   # ewm: x_t = Σ_{i=0->∞} (1-alpha)^i * x_{t-i} where w = 1-alpha in (0,1)
                                            # half-life: the time lag at which the exponential weights decays by one half.
                                            # pandas version: alpha = 1 - exp(-ln(2)/HL) <=> w = exp(-ln(2)/HL)
                                            # mathematical version: w^HL = w/2 <=> w = exp(-ln(2)/(HL-1))
                                            # analysis: won't make much of a difference
        self._vol_threshold = vol_threshold
        self._cut_nan_samples = cut_nan_samples
        self._symbols = symbols

    # Applied to pd.Series
    def _prepare_series(self, srs: pd.Series, eps: float=1e-6) -> pd.Series:
        '''
        To reduce the effect of extreme outliers, we winsorize all data to limit values 
        to be within 5 times its 252-day exponentially weighted moving standard deviation 
        from its exponentially weighted moving average (Tan, Roberts & Zohren, 2023, p.19).
        '''

        srs = srs.loc[(srs.notnull()) & (srs > eps)].copy() # eliminate all Null, NaN and values too close to the zero 
        ewm = srs.ewm(halflife=self._half_winsore)
        means = ewm.mean()  # 252-day exponentially weighted moving average
        stds = ewm.std()    # 252-day exponentially weighted moving standard deviation
        srs = np.minimum(srs, means + self._vol_threshold * stds)
        srs = np.maximum(srs, means - self._vol_threshold * stds)
        return srs

    def create_features(self) -> dict[pd.DataFrame]:
        '''
        Create a dictionary, where each key is an asset's name and each value is the pd.DataFrame
        of its features.
        '''
        features = {}
        for symbol in self._symbols: # for every asset

            # Create a dictionary with as keys the series' names and as values the pd.Series series
            prices = {'{}_srs'.format('close'): self._prices_df['{}_{}'.format(symbol, 'close')]}

            # Prepare every series in the dictionary by winsorising
            for key in prices.keys():
                prices[key] = self._prepare_series(prices[key])

            features_ = []
            for param, fc in zip(self._params, self._features):
                feature_creator = fc(**param)                       # calibrate the current feature creator with the parameters
                f = feature_creator.calc_features(**prices)         # calculate the features
                features_.append(f)                                 # add the features to the features' collection
            features_ = pd.concat(features_, axis=1)                # merge all features in the features' collection -> in a pd.DataFrame

            features_.set_index(self._prices_df.index)              # set index as original prices series
            features_ = features_.iloc[self._cut_nan_samples:-1]    # used to cut NaN samples after applying pd.Series.shift: recall that two of the features look 252 days back
            features[symbol] = features_                            # assign features pd.DataFrame to corresponding key in features dictionary

        return features