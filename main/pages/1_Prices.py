from gas_tank.utils import *

st.set_page_config(layout="centered")
check_session_initialization()

st.title('Upload prices')

uploaded_file = st.file_uploader("Upload excel file with prices.", help='Click to select a file to be uploaded.')

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.prices = import_prices(st.session_state.uploaded_file)
    st.write(st.session_state.prices)
elif st.session_state.uploaded_file is not None and uploaded_file is None:
    st.write(st.session_state.prices)
elif st.session_state.uploaded_file is None and uploaded_file is None:
    st.info('No file was uploaded.')