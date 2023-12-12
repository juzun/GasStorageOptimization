from utils import *

st.set_page_config(layout="wide")
check_session_initialization()

st.title('Optimize')

check_for_uploaded_prices()
check_for_initialized_storages()


st.button('Solve all', on_click=solve_all_button)

for (tab, storage) in zip(st.tabs(st.session_state.storage_labels), st.session_state.storages):
    with tab:
        st.button(f'Solve {storage.id}', on_click=solve_button, args=(storage,))
        
        if storage.solved:
            st.text(f'Storage {storage.id} optimized.\nObjective: {storage.objective}')            
        else:
            st.info('Not optimized yet.')