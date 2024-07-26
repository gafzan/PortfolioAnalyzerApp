"""yahoo_finance_data.py"""

import pandas as pd
from yahoo_fin import stock_info
from functools import reduce

MARKET_MAP = {"Chicago Board of Trade (CBOT)***": {"COUNTRY": "United States of America", "SUFFIX": ".CBT"},
              "Chicago Mercantile Exchange (CME)***": {"COUNTRY": "United States of America", "SUFFIX": ".CME"},
              "ICE Futures US": {"COUNTRY": "United States of America", "SUFFIX": ".NYB"},
              "New York Commodities Exchange (COMEX)***": {"COUNTRY": "United States of America", "SUFFIX": ".CMX"},
              "New York Mercantile Exchange (NYMEX)***": {"COUNTRY": "United States of America", "SUFFIX": ".NYM"},
              "Buenos Aires Stock Exchange (BYMA)": {"COUNTRY": "Argentina", "SUFFIX": ".BA"},
              "Vienna Stock Exchange": {"COUNTRY": "Austria", "SUFFIX": ".VI"},
              "Australian Stock Exchange (ASX)": {"COUNTRY": "Australia", "SUFFIX": ".AX"},
              "Cboe Australia": {"COUNTRY": "Australia", "SUFFIX": ".XA"},
              "Euronext Brussels": {"COUNTRY": "Belgium", "SUFFIX": ".BR"},
              "Sao Paolo Stock Exchange (BOVESPA)": {"COUNTRY": "Brazil", "SUFFIX": ".SA"},
              "Canadian Securities Exchange": {"COUNTRY": "Canada", "SUFFIX": ".CN"},
              "Cboe Canada": {"COUNTRY": "Canada", "SUFFIX": ".NE"},
              "Toronto Stock Exchange (TSX)": {"COUNTRY": "Canada", "SUFFIX": ".TO"},
              "TSX Venture Exchange (TSXV)": {"COUNTRY": "Canada", "SUFFIX": ".V"},
              "Santiago Stock Exchange": {"COUNTRY": "Chile", "SUFFIX": ".SN"},
              "Shanghai Stock Exchange": {"COUNTRY": "China", "SUFFIX": ".SS"},
              "Shenzhen Stock Exchange": {"COUNTRY": "China", "SUFFIX": ".SZ"},
              "Prague Stock Exchange Index": {"COUNTRY": "Czech Republic", "SUFFIX": ".PR"},
              "Nasdaq OMX Copenhagen": {"COUNTRY": "Denmark", "SUFFIX": ".CO"},
              "Egyptian Exchange Index (EGID)": {"COUNTRY": "Egypt", "SUFFIX": ".CA"},
              "Nasdaq OMX Tallinn": {"COUNTRY": "Estonia", "SUFFIX": ".TL"},
              "Cboe Europe": {"COUNTRY": "Europe", "SUFFIX": ".XD"}, "Euronext": {"COUNTRY": "Europe", "SUFFIX": ".NX"},
              "Nasdaq OMX Helsinki": {"COUNTRY": "Finland", "SUFFIX": ".HE"},
              "Euronext Paris": {"COUNTRY": "France", "SUFFIX": ".PA"},
              "Berlin Stock Exchange": {"COUNTRY": "Germany", "SUFFIX": ".BE"},
              "Bremen Stock Exchange": {"COUNTRY": "Germany", "SUFFIX": ".BM"},
              "Dusseldorf Stock Exchange": {"COUNTRY": "Germany", "SUFFIX": ".DU"},
              "Frankfurt Stock Exchange": {"COUNTRY": "Germany", "SUFFIX": ".F"},
              "Hamburg Stock Exchange": {"COUNTRY": "Germany", "SUFFIX": ".HM"},
              "Hanover Stock Exchange": {"COUNTRY": "Germany", "SUFFIX": ".HA"},
              "Munich Stock Exchange": {"COUNTRY": "Germany", "SUFFIX": ".MU"},
              "Stuttgart Stock Exchange": {"COUNTRY": "Germany", "SUFFIX": ".SG"},
              "Deutsche Boerse XETRA": {"COUNTRY": "Germany", "SUFFIX": ".DE"},
              "Collectable Indices": {"COUNTRY": "Global", "SUFFIX": ".REGA"},
              "Athens Stock Exchange (ATHEX)": {"COUNTRY": "Greece", "SUFFIX": ".AT"},
              "Hong Kong Stock Exchange (HKEX)*": {"COUNTRY": "Hong Kong", "SUFFIX": ".HK"},
              "Budapest Stock Exchange": {"COUNTRY": "Hungary", "SUFFIX": ".BD"},
              "Nasdaq OMX Iceland": {"COUNTRY": "Iceland", "SUFFIX": ".IC"},
              "Bombay Stock Exchange": {"COUNTRY": "India", "SUFFIX": ".BO"},
              "National Stock Exchange of India": {"COUNTRY": "India", "SUFFIX": ".NS"},
              "Indonesia Stock Exchange (IDX)": {"COUNTRY": "Indonesia", "SUFFIX": ".JK"},
              "Euronext Dublin": {"COUNTRY": "Ireland", "SUFFIX": ".IR"},
              "Tel Aviv Stock Exchange": {"COUNTRY": "Israel", "SUFFIX": ".TA"},
              "EuroTLX": {"COUNTRY": "Italy", "SUFFIX": ".TI"},
              "Italian Stock Exchange": {"COUNTRY": "Italy", "SUFFIX": ".MI"},
              "Tokyo Stock Exchange": {"COUNTRY": "Japan", "SUFFIX": ".T"},
              "Boursa Kuwait": {"COUNTRY": "Kuwait", "SUFFIX": ".KW"},
              "Nasdaq OMX Riga": {"COUNTRY": "Latvia", "SUFFIX": ".RG"},
              "Nasdaq OMX Vilnius": {"COUNTRY": "Lithuania", "SUFFIX": ".VS"},
              "Malaysian Stock Exchange": {"COUNTRY": "Malaysia", "SUFFIX": ".KL"},
              "Mexico Stock Exchange (BMV)": {"COUNTRY": "Mexico", "SUFFIX": ".MX"},
              "Euronext Amsterdam": {"COUNTRY": "Netherlands", "SUFFIX": ".AS"},
              "New Zealand Stock Exchange (NZX)": {"COUNTRY": "New Zealand", "SUFFIX": ".NZ"},
              "Oslo Stock Exchange": {"COUNTRY": "Norway", "SUFFIX": ".OL"},
              "Philippine Stock Exchange Indices": {"COUNTRY": "Philippines", "SUFFIX": ".PS"},
              "Warsaw Stock Exchange": {"COUNTRY": "Poland", "SUFFIX": ".WA"},
              "Euronext Lisbon": {"COUNTRY": "Portugal", "SUFFIX": ".LS"},
              "Qatar Stock Exchange": {"COUNTRY": "Qatar", "SUFFIX": ".QA"},
              "Bucharest Stock Exchange": {"COUNTRY": "Romania", "SUFFIX": ".RO"},
              "Singapore Stock Exchange (SGX)": {"COUNTRY": "Singapore", "SUFFIX": ".SI"},
              "Johannesburg Stock Exchange": {"COUNTRY": "South Africa", "SUFFIX": ".JO"},
              "Korea Stock Exchange": {"COUNTRY": "South Korea", "SUFFIX": ".KS"},
              "KOSDAQ": {"COUNTRY": "South Korea", "SUFFIX": ".KQ"},
              "Madrid SE C.A.T.S.": {"COUNTRY": "Spain", "SUFFIX": ".MC"},
              "Saudi Stock Exchange (Tadawul)": {"COUNTRY": "Saudi Arabia", "SUFFIX": ".SAU"},
              "Nasdaq OMX Stockholm": {"COUNTRY": "Sweden", "SUFFIX": ".ST"},
              "Swiss Exchange (SIX)": {"COUNTRY": "Switzerland", "SUFFIX": ".SW"},
              "Taiwan OTC Exchange": {"COUNTRY": "Taiwan", "SUFFIX": ".TWO"},
              "Taiwan Stock Exchange (TWSE)": {"COUNTRY": "Taiwan", "SUFFIX": ".TW"},
              "Stock Exchange of Thailand (SET)": {"COUNTRY": "Thailand", "SUFFIX": ".BK"},
              "Borsa İstanbul": {"COUNTRY": "Turkey", "SUFFIX": ".IS"},
              "Dubai Financial Market": {"COUNTRY": "United Arab Emirates", "SUFFIX": ".AE"},
              "Cboe UK": {"COUNTRY": "United Kingdom", "SUFFIX": ".XC"},
              "London Stock Exchange": {"COUNTRY": "United Kingdom", "SUFFIX": ".L"},
              # "London Stock Exchange": {"COUNTRY": "United Kingdom", "SUFFIX": ".IL"},
              "Caracas Stock Exchange": {"COUNTRY": "Venezuela", "SUFFIX": ".CR"}, }


def get_raw_stock_data(tickers: {str, list}, start_date: str = None, end_date: str = None,
                       return_as_df: bool = False) -> {pd.DataFrame, dict}:
    """
    Returns a DataFrame or dict with OHLC and volume data downloaded from Yahoo Finance
    :param tickers: str or list of str
    :param start_date: str
    :param end_date: str
    :param return_as_df: bool If True returns a DataFrame otherwise dict
    :return: DataFrame or dict
    """
    # convert ticker to list if applicable
    if not isinstance(tickers, list):
        tickers = [tickers]

    # pull price data for all tickers
    data = {t: stock_info.get_data(t, start_date=start_date, end_date=end_date) for t in tickers}

    if return_as_df:
        data = reduce(lambda x, y: x._append(y), data.values())

    return data


def _get_price_data(tickers: {str, list}, data_fld: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame with price data
    :param tickers: str or list of str
    :param data_fld: str data field
    :param start_date: str
    :param end_date: str
    :return: DataFrame
    """
    # handle the data field input
    data_fld = data_fld.lower()
    eligible_flds = ['open', 'high', 'low', 'close', 'adjclose', 'volume']
    if data_fld not in eligible_flds:
        raise ValueError(f"'{data_fld}' needs to be one of '%s'" % "', '".join(eligible_flds))

    raw_df = get_raw_stock_data(tickers=tickers, start_date=start_date, end_date=end_date, return_as_df=True)

    # pivot the raw data
    df = pd.pivot_table(raw_df.reset_index(), values=data_fld, index=['index'], columns=['ticker'])
    return df


def get_open_price_data(tickers: {str, list}, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame with open price data
    :param tickers: str or list of str
    :param start_date: str
    :param end_date: str
    :return: DataFrame
    """
    return _get_price_data(tickers=tickers, data_fld='open', start_date=start_date, end_date=end_date)


def get_high_price_data(tickers: {str, list}, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame with high price data
    :param tickers: str or list of str
    :param start_date: str
    :param end_date: str
    :return: DataFrame
    """
    return _get_price_data(tickers=tickers, data_fld='high', start_date=start_date, end_date=end_date)


def get_low_price_data(tickers: {str, list}, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame with low price data
    :param tickers: str or list of str
    :param start_date: str
    :param end_date: str
    :return: DataFrame
    """
    return _get_price_data(tickers=tickers, data_fld='low', start_date=start_date, end_date=end_date)


def get_close_price_data(tickers: {str, list}, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame with close price data
    :param tickers: str or list of str
    :param start_date: str
    :param end_date: str
    :return: DataFrame
    """
    return _get_price_data(tickers=tickers, data_fld='close', start_date=start_date, end_date=end_date)


def get_adjclose_price_data(tickers: {str, list}, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame with adjclose price data
    :param tickers: str or list of str
    :param start_date: str
    :param end_date: str
    :return: DataFrame
    """
    return _get_price_data(tickers=tickers, data_fld='adjclose', start_date=start_date, end_date=end_date)
