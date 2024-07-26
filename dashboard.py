"""dashboard.py"""
import math

import streamlit as st
import uuid
import pandas as pd

from yahoo_finance_data import get_adjclose_price_data
from yahoo_finance_data import MARKET_MAP

if "ticker_rows" not in st.session_state:
    st.session_state["ticker_rows"] = []


# initializing the portfolio (will be a list of dict)
portfolio = []


def add_ticker_row() -> None:
    """
    Creates a new UUID4 string and appends it to the session state variable called rows
    :return: None
    """
    element_id = uuid.uuid4()
    st.session_state["ticker_rows"].append(str(element_id))
    return


def remove_ticker_row(row_id) -> None:
    """
    removes a given row_id string from the session state variable
    :param row_id:
    :return: None
    """
    st.session_state["ticker_rows"].remove(str(row_id))
    return


def generate_ticker_row(row_id, num_input_label: str) -> dict:
    """
    Generates a row with a ticker, quantity, market and delete button column
    :param row_id:
    :param num_input_label: str
    :return: dict
    """
    row_container = st.empty()
    row_columns = row_container.columns((2, 2, 2, 1))
    row_name = row_columns[0].text_input("Ticker", key=f"txt_{row_id}").upper()
    row_qty = row_columns[1].number_input(num_input_label, step=1, key=f"nbr_{row_id}")
    row_mkt = row_columns[2].selectbox("Market (optional)", [''] + list(MARKET_MAP.keys()), key=f'mkt_{row_id}')
    if len(row_mkt):
        suffix = MARKET_MAP[row_mkt]['SUFFIX']
    else:
        suffix = ''
    row_columns[3].button("🗑️", key=f"del_{row_id}", on_click=remove_ticker_row, args=[row_id])
    return {"ticker": row_name + suffix, "qty": row_qty}


def calculation_test():
    # get the tickers
    tickers = [d['ticker'] for d in portfolio if len(d['ticker'])]
    px_df = get_adjclose_price_data(tickers=tickers)
    px_df.dropna(inplace=True)
    vol = px_df.pct_change().std() * math.sqrt(252)
    st.write(vol)
    return


# this one selects what page you want to look at
dashboard_option = st.sidebar.selectbox("Select dashboard:",
                                        options=['Correlation',
                                                 'Beta'])

# widgets for the sidebar
with st.sidebar.expander('Portfolio'):

    weight_or_shares_option = st.radio('Weight or shares', ['Weight', 'Shares'])

    if len(st.session_state["ticker_rows"]) == 0:
        add_ticker_row()
        add_ticker_row()
        add_ticker_row()

    for ticker_row in st.session_state['ticker_rows']:
        ticker_row_data = generate_ticker_row(ticker_row, num_input_label=weight_or_shares_option)
        portfolio.append(ticker_row_data)

    st.button('New position', on_click=add_ticker_row)

    # display
    if len(portfolio):
        st.subheader('Portfolio')
        df = pd.DataFrame(portfolio)
        df.rename(columns={'ticker': 'Ticker', 'qty': weight_or_shares_option}, inplace=True)
        st.dataframe(data=df, use_container_width=True)


st.button('Calculate', on_click=calculation_test)
