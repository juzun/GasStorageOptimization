from utils import *

st.set_page_config(layout="wide")
check_session_initialization()

st.title('Initialize storages')

check_for_uploaded_prices()


if st.button('Initialize storage 1'):
    st.session_state.storages.append(initialize_storage1(st.session_state.prices))
    update_storage_labels()

if st.button('Initialize storage 2'):
    st.session_state.storages.append(initialize_storage2(st.session_state.prices))
    update_storage_labels()

if st.button('Initialize storage'):
    st.session_state.storages.append(initialize_storage())
    update_storage_labels()

for index, storage in enumerate(st.session_state.storages):
    col1, col2, col3 = st.columns((4,2,1))
    col1.write(f'{storage.id}')
    message = 'Optimized'  if storage.solved else 'Not optimized'
    col2.write(message)
    if col3.button('Delete', key=f'del{index}'):
        del st.session_state.storages[index]
        update_storage_labels()
