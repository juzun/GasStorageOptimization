from utils import check_session_initialization, import_prices, authentication_check
import streamlit as st

st.set_page_config(layout="wide")

authentication_check()
check_session_initialization()

st.title("Upload prices")
st.write("---")

col1, col2 = st.columns((3, 2))

uploaded_file = col1.file_uploader(
    "Upload excel file with prices.", help="Click to select a file to be uploaded."
)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.prices = import_prices(st.session_state.uploaded_file)
    col1.success("The file was successfuly uploaded")
    col2.write(st.session_state.prices)
elif st.session_state.uploaded_file is not None and uploaded_file is None:
    col2.write(st.session_state.prices)
elif st.session_state.uploaded_file is None and uploaded_file is None:
    col2.info("No file was uploaded.")
