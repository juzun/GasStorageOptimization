from gas_tank.utils import *

check_session_initialization()

st.set_page_config(layout="wide")
st.title('Export')

check_for_uploaded_prices()
check_for_initialized_storages()
check_for_solved_storages()


col1, col2 = st.columns(2)
col1.write('Daily operations')
daily_format = {key: '{:.0f}' for key in st.session_state.storages[0].daily_export.columns}
daily_format['Stav %'] = '{:.2%}'
col1.dataframe(st.session_state.storages[0].daily_export.style.format(daily_format))

col2.write('Monthly operations')
monthly_format = {key: '{:.0f}' for key in st.session_state.storages[0].monthly_export.columns}
monthly_format['Stav %'] = '{:.2%}'
col2.dataframe(st.session_state.storages[0].monthly_export.style.format(daily_format))

st.download_button(
    label='Download .xlsx export',
    data=export_to_xlsx(st.session_state.storages[0]),
    file_name=f'{st.session_state.storages[0].id}.xlsx',
    mime='application/vnd.ms-excel',
)
