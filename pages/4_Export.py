from gas_tank.utils import *

st.set_page_config(layout="wide")
check_session_initialization()

st.title('Export')

check_for_uploaded_prices()
check_for_initialized_storages()
check_for_solved_storages()



# Total export
collect_storages()

col1, col2 = st.columns((3,2))
col1.write('Total daily operations')
daily_format = {key: '{:.0f}' for key in st.session_state.total_daily_export.columns}
daily_format['Stav %'] = '{:.2%}'
col1.dataframe(st.session_state.total_daily_export.style.format(daily_format), use_container_width=True)

col2.write('Total monthly operations')
monthly_format = {key: '{:.0f}' for key in st.session_state.total_monthly_export.columns}
monthly_format['Stav %'] = '{:.2%}'
col2.dataframe(st.session_state.total_monthly_export.style.format(daily_format), use_container_width=True)

_, last_col = st.columns([3,1])
with last_col:
    st.download_button(
        label='Download total export',
        data=total_export_to_xlsx(),
        file_name='total_export.xlsx',
        mime='application/vnd.ms-excel',
    )


# Export of storages

storages_labels = [storage.id for storage in st.session_state.storages]
for (tab, storage) in zip(st.tabs(st.session_state.storage_labels), st.session_state.storages):
    with tab:
        st.subheader(storage.id)

        if storage.solved:
            col1, col2 = st.columns((3,2))
            col1.write('Daily operations')
            daily_format = {key: '{:.0f}' for key in storage.daily_export.columns}
            daily_format['Stav %'] = '{:.2%}'
            col1.dataframe(storage.daily_export[['Rok','M','W/I','Stav','Stav %','Max C']].style.format(daily_format), use_container_width=True)

            col2.write('Monthly operations')
            monthly_format = {key: '{:.0f}' for key in st.session_state.storages[0].monthly_export.columns}
            monthly_format['Stav %'] = '{:.2%}'
            col2.dataframe(storage.monthly_export[['Rok','M','W/I','Stav','Stav %']].style.format(daily_format), use_container_width=True)

            _, last_col = st.columns([4,1])
            with last_col:
                st.download_button(
                    label='Download export',
                    data=export_to_xlsx(st.session_state.storages[0]),
                    file_name=f'{storage.id}.xlsx',
                    mime='application/vnd.ms-excel',
                )
        else:
            st.info('Not optimized yet.')