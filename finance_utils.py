"""finance_utils.py"""

import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS
import plotly.express as px


def portfolio_backtest(px_data_df: pd.DataFrame, weight_map: dict, initial_value: float = 100.0,
                       ticker_currency_map: dict = None):
    """
    Calculates a backtest of a portfolio assuming that weights are rebalanced every day
    :param px_data_df: DataFrame, index = dates, columns = tickers
    :param weight_map: dict ticker: weights
    :param initial_value: float
    :param ticker_currency_map: dict
    :return: DataFrame
    """
    px_data_df = px_data_df.copy()
    # convert underlying to different currencies if applicable
    if ticker_currency_map:
        for ticker, ccy_ticker in ticker_currency_map.items():
            px_data_df[ticker] *= px_data_df[ccy_ticker].values

    # select the data for the tickers
    px_data_df = px_data_df.loc[:, weight_map.keys()].copy()

    # normalize the weights (if all are 100% it will be an equal weight i.e. 1/N)
    weight_s = pd.Series(weight_map)
    weight_s /= weight_s.sum()

    # calculate the back test as the cumulative products of the sum of weighted daily returns
    port_bt_df = pd.DataFrame(
        initial_value * (1 + (px_data_df.pct_change() * weight_s).sum(axis=1)).cumprod(),
        columns=['Portfolio'],
    )
    return port_bt_df


def rolling_beta(price_return_df: pd.DataFrame, benchmark_col_name: str, window: int) -> {pd.DataFrame, dict}:
    """
    Returns a DataFrame or dict performing OLS for Beta, returning the beta (slope), p-values, adj. r-squared
    :param price_return_df: DataFrame
    :param benchmark_col_name: str
    :param window: int
    :return: dict or DataFrame
    """

    # exogenious variables
    exog_var = price_return_df[[benchmark_col_name]]
    exog_var.loc[:, 'const'] = 1.0

    tickers_exl_bench = list(price_return_df)
    try:
        tickers_exl_bench.remove(benchmark_col_name)
    except ValueError:
        pass

    result = {}
    for t in tickers_exl_bench:
        model = RollingOLS(endog=price_return_df[t], exog=exog_var, window=window)
        rres = model.fit()
        result[t] = pd.concat(
            [
                rres.params,
                pd.DataFrame(
                    data=rres.pvalues,
                    columns=[f'p_value_{c}' for c in rres.params.columns],
                    index=rres.params.index
                ),
                rres.rsquared_adj.to_frame('rsquared_adj')
            ],
            axis=1
        )

    if len(result) > 1:
        return result
    else:
        return result[list(result.keys())[0]]


if __name__ == '__main__':
    from yahoo_finance_data import get_adjclose_price_data
    from yahoo_finance_data import get_currency_ticker_map
    benchmark = 'VTI'
    window = 100
    #
    # tickers = ['^STOXX', 'VGK', 'MC.PA']
    #
    # target_ccy = 'EUR'
    # fx_ticker_map = get_currency_ticker_map(ticker_currency_target_map={
    #     '^STOXX': target_ccy,
    #     'VGK': target_ccy
    # })
    # port_weight = {'VGK': 1.}

    tickers = ['ERIC', 'ERIC-B.ST']
    target_ccy = 'SEK'
    fx_ticker_map = get_currency_ticker_map(ticker_currency_target_map={
        'ERIC': target_ccy,
        'ERIC-B.ST': target_ccy
    })
    print(fx_ticker_map)

    tickers.extend(
        [fx_ticker for fx_ticker in list(set(fx_ticker_map.values()))]
    )

    tickers = [t.upper() for t in tickers]

    from yahoo_finance_data import get_close_price_data
    df = get_adjclose_price_data(tickers=tickers)
    df.dropna(inplace=True)

    # eur_vgk = portfolio_backtest(px_data_df=df, weight_map=port_weight, ticker_currency_map=fx_ticker_map)

    # df = pd.concat([df, eur_vgk], axis=1)

    # fig = px.line(data_frame=df)
    # fig.show()

    df.to_clipboard()

    # beta = rolling_beta(price_return_df=px_ret_df, benchmark_col_name=benchmark, window=window)
    # print(beta)

    # print(beta[::-1])

