from gas_tank.utils import *

check_session_initialization()

st.set_page_config(layout="wide")
st.title('Graphs')

check_for_uploaded_prices()
check_for_initialized_storages()
check_for_solved_storages()

st.plotly_chart(get_graph(st.session_state.storages[0]), use_container_width=True)

st.download_button(
    label='Download graph',
    data=get_graph(st.session_state.storages[0]).to_html(),
    file_name=f'{st.session_state.storages[0].id}.html',
    mime='text/html',
)