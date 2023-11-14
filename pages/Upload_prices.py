import streamlit as st
from gas_tank.utils import import_prices


uploaded_file = st.file_uploader("Upload .xlsx file with prices.")

if uploaded_file is not None:
    prices = import_prices(uploaded_file)
    st.write(prices)
    st.session_state.prices = prices
else:
    st.info('No file was uploaded.')