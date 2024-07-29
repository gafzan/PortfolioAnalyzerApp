"""dashboard.py"""

import streamlit as st
import uuid
import pandas as pd
import datetime
import plotly.express as px

from yahoo_finance_data import get_adjclose_price_data
from yahoo_finance_data import get_close_price_data
from yahoo_finance_data import MARKET_MAP
from yahoo_finance_data import get_currency_ticker_map

from finance_utils import portfolio_backtest
from finance_utils import rolling_beta

st.set_page_config(layout="wide")

if "portfolio_ticker_rows" not in st.session_state:
    st.session_state["portfolio_ticker_rows"] = []

if "benchmark_ticker_rows" not in st.session_state:
    st.session_state["benchmark_ticker_rows"] = []

# data to be used in the app
CURRENCY = ['USD', 'EUR', 'JPY', 'GBP', 'HKD', 'AUD', 'NZD', 'CAD', 'SEK', 'CHF']

RETURN_CALCULATION_CONFIG = {
    'Daily Returns': {
        'label': 'Trading days',
        'default': 252,
        'daily_lag': 1
    },
    'Weekly Returns': {
        'label': 'weeks',
        'default': 50,
        'daily_lag': 5
    },
    'Monthly Returns': {
        'label': 'months',
        'default': 12,
        'daily_lag': 20
    },
    'Yearly Returns': {
        'label': 'years',
        'default': 3,
        'daily_lag': 252
    }
}

# initializing the position container for the portfolio and benchmark positions
table_positions_map = {
    'portfolio': [],
    'benchmark': []
}


def add_ticker_row(table_name: str) -> None:
    """
    Creates a new UUID4 string and appends it to the session state variable called rows
    :return: None
    """
    element_id = uuid.uuid4()
    st.session_state[f"{table_name}_ticker_rows"].append(str(element_id))
    return


def remove_ticker_row(row_id, table_name: str) -> None:
    """
    removes a given row_id string from the session state variable
    :param row_id:
    :param table_name: str
    :return: None
    """
    st.session_state[f"{table_name}_ticker_rows"].remove(str(row_id))
    return


def generate_ticker_row(row_id, num_input_label: str, table_name: str) -> dict:
    """
    Generates a row with a ticker, quantity, market and delete button column
    :param row_id:
    :param num_input_label: str
    :param table_name: str
    :return: dict
    """
    row_container = st.empty()
    ticker_row_columns = row_container.columns((2, 2, 3, 2, 1))
    row_tick = ticker_row_columns[0].text_input("Ticker", key=f"txt_{row_id}",
                                                help="Insert Yahoo Finance tickers. Tickers for indices starts with "
                                                     "'^'. For example, to get S&P 500 type '^SPX'").upper()
    if num_input_label == 'Weight (%)':
        row_qty = ticker_row_columns[1].slider(num_input_label, min_value=0, max_value=100, value=100,
                                               key=f"nbr_{row_id}",
                                               help='Weights will later be normalized such that the '
                                                    'sum of weights equals 100%')
    else:
        row_qty = ticker_row_columns[1].number_input(num_input_label, min_value=0, step=1, key=f"nbr_{row_id}")
    row_mkt = ticker_row_columns[2].selectbox("Market", [''] + list(MARKET_MAP.keys()), key=f'mkt_{row_id}',
                                              help="(Optional) For tickers outside the U.S. you likely need to add a "
                                                   "'suffix' after the ticker, representing the exchange the security "
                                                   "is listed on. You can either type the suffix directly in the "
                                                   "'Ticker' input or look up the Exchange for your security.")
    if len(row_mkt) and len(row_tick):
        row_tick += MARKET_MAP[row_mkt]['SUFFIX']

    row_fx = ticker_row_columns[3].selectbox("FX", [''] + CURRENCY, key=f'fx_{row_id}',
                                             help="(Optional) Choose a currency to convert to. For example, 'EUR' will "
                                                  "convert the daily prices for the chosen security to Euros")
    ticker_row_columns[4].button("🗑️", key=f"del_{row_id}", on_click=remove_ticker_row, args=[row_id, table_name])
    return {"ticker": row_tick, "qty": row_qty, "fx": row_fx}


def get_table_tickers(table_name: str) -> list:
    """
    Returns a list of tickers for each position in either the portfolio or benchmark table
    :param table_name: str
    :return: list
    """
    positions = table_positions_map[table_name]
    tickers = [position['ticker'] for position in positions if len(position['ticker'])]
    return tickers


def get_ticker_weights_map(table_name: str) -> dict:
    """
    Returns a dict with tickers as keys and the corresponding weight as values
    :param table_name: str
    :return: dict
    """
    positions = table_positions_map[table_name]
    result = {position['ticker']: position['qty'] for position in positions if len(position['ticker'])}
    return result


def get_ticker_currency_symbol_map(table_name: str) -> dict:
    """
    Returns a dict with tickers as keys and the corresponding Yahoo Finance currency ticker to use for a specified
    target currency.
    For example if for 'NVDA' user has specified FX = 'EUR', the corresponding Yahoo Finance currency ticker to convert
    the USD denominated NVDA prices is 'EUR=X'
    :param table_name: str
    :return: dict
    """
    positions = table_positions_map[table_name]
    ticker_target_ccy_map = {position['ticker']: position['fx'] for position in positions if
                             (len(position['ticker']) and len(position['fx']))}
    result = get_currency_ticker_map(ticker_currency_target_map=ticker_target_ccy_map)
    return result


@st.cache_data
def get_price_df(tickers: list) -> pd.DataFrame:
    """
    Download daily prices from Yahoo Finance for the specified tickers
    :param tickers: list of str
    :return: DataFrame
    """
    if price_type == 'Close':
        px_df = get_close_price_data(tickers=tickers)
    elif price_type == 'Adj. Close':
        px_df = get_adjclose_price_data(tickers=tickers)
    else:
        raise ValueError(f"'{price_type}' is not a recognized price type")
    return px_df


def get_price_return_df(px_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with price return data
    :param px_df: DataFrame
    :return: DataFrame
    """
    return_basis_lag = RETURN_CALCULATION_CONFIG[px_return_basis]['daily_lag']
    if allow_return_overlap:
        px_return_df = px_df.pct_change(return_basis_lag)
    else:
        px_return_df = px_df.sort_index(ascending=False)[::-return_basis_lag].sort_index(ascending=True).pct_change()

    px_return_df.dropna(inplace=True)
    return px_return_df


def make_scatter_plot(px_return_df: pd.DataFrame) -> None:
    """
    Makes a scatter plot with a OLS trend line
    :param px_return_df:
    :return:
    """
    px_return_scatter_data_df = px_return_df.copy()
    px_return_scatter_data_df *= 100.0
    px_return_scatter_data_df.columns = [f'{c} (%)' for c in list(px_return_scatter_data_df)]
    px_return_scatter_data_df = px_return_scatter_data_df.reset_index().rename(columns={'index': 'Date'})
    px_return_scatter_data_df['Date'] = px_return_scatter_data_df['Date'].dt.date
    fig = px.scatter(px_return_scatter_data_df,
                     x='Benchmark (%)',
                     y='Portfolio (%)',
                     title='Regression Beta',
                     hover_data=['Date'],
                     trendline="ols",
                     trendline_color_override='red')
    st.plotly_chart(fig, key="beta_scatter", on_select="rerun")
    return


def calculate_and_display_beta(px_return_df: pd.DataFrame) -> None:
    """
    Calculates total and rolling Beta and displays the result in a scatter plot as well as a line chart
    :param px_return_df: DataFrame
    :return: None
    """
    # calculate the total historical beta
    total_hist_beta = rolling_beta(price_return_df=px_return_df.loc[start_date: end_date, :],
                                   benchmark_col_name='Benchmark',
                                   window=px_return_df.loc[start_date: end_date, :].shape[0]).iloc[-1, :]

    # show the results: Beta, R2 and p-value
    result_columns = st.empty().columns(3)
    result_columns[0].metric('Beta', round(total_hist_beta['Benchmark'], 2))
    result_columns[1].metric('Adj. R squared', f"{round(100 * total_hist_beta['rsquared_adj'], 2)}%")
    result_columns[2].metric('p-value (Beta)', f"{round(100 * total_hist_beta['p_value_Benchmark'], 2)}%")

    # make scatter plot
    make_scatter_plot(px_return_df=px_return_df)

    # make rolling beta correlation line plot
    beta_roll = rolling_beta(price_return_df=px_return_df,
                             benchmark_col_name='Benchmark',
                             window=rolling_config)

    # select based on date range if specified in the configuration panel
    beta_roll = beta_roll.loc[start_date: end_date, :].copy()
    beta_roll = beta_roll.reset_index().rename(columns={'index': 'Date'})
    beta_roll['Date'] = beta_roll['Date'].dt.date
    beta_roll.rename(
        columns=
        {
            'Benchmark': 'Beta',
            'const': 'Intercept',
            'p_value_Benchmark': 'p-value (Beta)',
            'p_value_const': 'p-value (Intercept)',
            'rsquared_adj': 'Adj. R squared'
        },
        inplace=True
    )
    fig = px.line(beta_roll,
                  x='Date',
                  y='Beta',
                  title='Rolling Beta',
                  hover_data=['p-value (Beta)', 'Adj. R squared'])
    st.plotly_chart(fig, key="beta_line_plot", on_select="rerun")
    return


def perform_calculation() -> None:
    """
    Gathers all the meta data for each position from the portfolio and benchmark and then calculates the total
    historical and rolling beta. Resulting betas are displayed in a scatter and line plot
    :return: None
    """
    # get the tickers for the portfolio and benchmark
    portfolio_tickers = get_table_tickers(table_name='portfolio')
    benchmark_tickers = get_table_tickers(table_name='benchmark')
    portfolio_tickers_fx_map = get_ticker_currency_symbol_map(table_name='portfolio')
    benchmark_tickers_fx_map = get_ticker_currency_symbol_map(table_name='benchmark')
    fx_tickers = list(portfolio_tickers_fx_map.values()) + list(benchmark_tickers_fx_map.values())
    tickers = portfolio_tickers + benchmark_tickers + fx_tickers
    tickers = list(set(tickers))  # unique tickers only

    # check so that there are enough tickers
    if not len(portfolio_tickers):
        st.write("Portfolio is empty! (Click 'Portfolio' in the sidebar)")
        return
    elif not len(benchmark_tickers):
        st.write("No benchmark selected! (Click 'Benchmark' below)")
        return

    # download the prices for each ticker
    # sort the tickers so that if the tickers are the same there will be no download of data (cache data)
    tickers.sort()
    px_df = get_price_df(tickers=tickers)

    # forward fill all nan for the FX tickers
    for fx_ticker in fx_tickers:
        px_df[fx_ticker].fillna(method='ffill', inplace=True)
    px_df.dropna(inplace=True)

    # calculate the backtested portfolio
    portfolio_ticker_weight_map = get_ticker_weights_map(table_name='portfolio')
    bt_df = portfolio_backtest(
        px_data_df=px_df,
        weight_map=portfolio_ticker_weight_map,
        ticker_currency_map=portfolio_tickers_fx_map)

    # calculate the backtested benchmark
    benchmark_ticker_weight_map = get_ticker_weights_map(table_name='benchmark')
    benchmark_bt_df = portfolio_backtest(
        px_data_df=px_df,
        weight_map=benchmark_ticker_weight_map,
        ticker_currency_map=benchmark_tickers_fx_map)
    benchmark_bt_df.columns = ['Benchmark']

    # merge the portfolio with the benchmark back test as the intersection of the two calendars
    bt_df = pd.merge(bt_df, benchmark_bt_df, left_index=True, right_index=True)

    # clean the results before calculating returns
    bt_df.dropna(inplace=True)

    # calculate price returns
    px_return_df = get_price_return_df(px_df=bt_df)

    # scatter plot of the beta measured over the entire selected period
    calculate_and_display_beta(px_return_df=px_return_df)
    return


def setup_position_widgets(table_name: str, input_type: str) -> None:
    """
    sets up input rows for tickers to be added to a portfolio
    :param table_name: str
    :param input_type: str
    :return: None
    """
    if input_type not in ['Weight (%)', 'Shares']:
        raise ValueError(f"'{input_type}' not recognized")

    if len(st.session_state[f"{table_name}_ticker_rows"]) == 0:
        add_ticker_row(table_name=table_name)

    for ticker_row in st.session_state[f"{table_name}_ticker_rows"]:
        ticker_row_data = generate_ticker_row(ticker_row,
                                              num_input_label=input_type,
                                              table_name=table_name)
        table_positions_map[table_name].append(ticker_row_data)

    # click to add new rows
    st.button('New position', on_click=add_ticker_row, kwargs={'table_name': table_name}, key=f"new_pos_{table_name}")
    display_positions(table_name=table_name, input_type=input_type)
    return


def display_positions(table_name: str, input_type: str) -> None:
    """
    Displays the positions either as a table or a pie chart with weights depending on the input_type
    :param table_name: str
    :param input_type: str
    :return: None
    """
    tickers = get_table_tickers(table_name=table_name)
    if len(tickers):
        if input_type == 'Shares':
            df = pd.DataFrame(table_positions_map[table_name])
            df.rename(columns={'ticker': 'Ticker', 'qty': input_type}, inplace=True)
            st.dataframe(data=df, use_container_width=True)
        else:
            # display a pie chart with normalized weights to sum to 100%
            df = pd.DataFrame(table_positions_map[table_name])
            df.rename(columns={'ticker': 'Ticker', 'qty': input_type}, inplace=True)
            df = df.loc[df['Ticker'].isin(tickers), :]
            df.loc[:, input_type] /= df.loc[:, input_type].sum()  # normalize the weights
            fig = px.pie(data_frame=df,
                         values=input_type,
                         names='Ticker',
                         title=f'{table_name.title()} allocation')
            st.plotly_chart(fig)
    return


# ----------------------------------------------------------------------------------------------------------------------
# Main page
st.title('Beta')
st.write('This tool allows you to calculate regression Betas for a specified portfolio and a benchmark.\nThe benchmark '
         'can either be a single or a weighted portfolio of underlyings.')
with st.expander('Implementation steps'):
    st.write(
        "1. Create a portfolio in the left sidebar\n\n2. Setup a benchmark by clicking 'Benchmark' below\n\n3. (Optional) "
        "Adjust the Beta calculation under 'Model configuration'\n\n4. Click 'Calculate beta'"
    )
st.button(f'Calculate beta', on_click=perform_calculation)

price_type = st.sidebar.radio('Price', ['Close', 'Adj. Close'], help="Adjusted close is the closing price after "
                                                                     "adjustments for all applicable splits and dividend "
                                                                     "distributions.")

# ----------------------------------------------------------------------------------------------------------------------
# widgets for the sidebar
with st.sidebar.expander('Portfolio'):
    # weight_or_shares_option = st.radio('Weight or shares', ['Weight (%)', 'Shares'])
    weight_or_shares_option = 'Weight (%)'

    setup_position_widgets(table_name='portfolio', input_type=weight_or_shares_option)

# ----------------------------------------------------------------------------------------------------------------------
# Benchmark for Beta
with st.expander('Benchmark'):
    setup_position_widgets(table_name='benchmark', input_type='Weight (%)')

# ----------------------------------------------------------------------------------------------------------------------
# model configuration for the beta calculation like return observation and window for the rolling calculation
with st.expander(f'Model configuration'):
    config_row_columns = st.empty().columns((2, 2))
    px_return_basis = config_row_columns[0].selectbox('Return basis',
                                                      [k for k, _ in RETURN_CALCULATION_CONFIG.items()])
    allow_return_overlap = config_row_columns[1].toggle('Allow overlapping returns', value=False)

    if allow_return_overlap:
        default_window = 252
    else:
        default_window = RETURN_CALCULATION_CONFIG[px_return_basis]['default']

    rolling_config = st.number_input(f"{RETURN_CALCULATION_CONFIG[px_return_basis]['label'].title()} (Rolling Beta)",
                                     value=default_window,
                                     min_value=0)
    row_columns = st.empty().columns((2, 2))
    start_date = row_columns[0].date_input('Start date (optional)', value=None,
                                           min_value=datetime.datetime(year=1900, month=1, day=1))
    end_date = row_columns[1].date_input('End date (optional)', value=None,
                                         min_value=datetime.datetime(year=1900, month=1, day=1))

# ----------------------------------------------------------------------------------------------------------------------
# explanation of the model and various results in a wiki page
with st.expander("Wiki"):
    st.header('Beta')
    st.write("Beta (β) is a statistic that measures the estimated increase or decrease of an individual stock price or "
             "portfolio proportion to movements of a benchmark. When the benchmark is a broad market index, Beta can be "
             "refered to an asset's non-diversifiable risk, systematic risk, or market risk.")
    st.write(r"""
    The beta ${\displaystyle \beta}$ of portfolio ${P}$ observed on 𝑡 occasions with respect to a benchmark ${B}$, is 
    obtained by a linear regression of the rate of return ${\displaystyle R_{P,t}}$ of portfolio ${P}$ on the rate of 
    return of ${\displaystyle R_{B,t}}$ of benchmark ${B}$.
    $$
    {\displaystyle R_{P,t}=\alpha+\beta \cdot R_{B,t}+\varepsilon _{t}}
    $$
    where ${\displaystyle \varepsilon _{t}}$ is an unbiased error term whose squared error should be minimized.
    
    The ordinary least squares solution is:
    $$
    {\displaystyle \beta ={\frac {\operatorname {Cov} (R_{P},R_{B})}{\operatorname {Var} (R_{B})}},}
    $$
    where ${\displaystyle \operatorname {Cov} }$ and ${\displaystyle \operatorname {Var} }$ are the covariance and 
    variance operators.
    """)

    st.header(r"${\displaystyle R ^{2}}$")
    st.write(
        """
        Statistical measure representing the proportion of the variance in portfolio returns that 
        is explained by the benchmark returns in the regression. Higher ${\displaystyle R ^{2}}$ means a higher 
        percentage of the variance in returns can be explained by the provided factors.
        """
    )
    st.header(r"${\displaystyle p}$-value")
    st.write(
        r"""
        When estimating betas, one is attempting to reject the "null hypothesis" ${\displaystyle H_{0}}$ that states 
        that the beta is zero
        $$
        {\displaystyle H_{0}: \beta = 0}
        $$
        in favor of the "alternative hypothesis" ${\displaystyle H_{a}}$
        $$
        {\displaystyle H_{a}: \beta \neq 0}
        $$
        The ${\displaystyle p}$-value represents the probability of obtaining an estimate of beta 
        ${\displaystyle \hat{\beta}}$ while still not rejecting ${\displaystyle H_{0}}$. For example, assuming we have 
        an estimate ${\displaystyle \hat{\beta}}$ with a ${\displaystyle p}$-value = 5%. This means that it is very 
        unlikely that we observe our estimate as ${\displaystyle \hat{\beta}}$ while still assuming that ${\displaystyle H_{0}}$ is correct. 
        We could instead reject the ${\displaystyle H_{0}}$ with 5% significance level.
        """
    )
