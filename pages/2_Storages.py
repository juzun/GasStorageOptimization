from gas_tank.utils import *

check_session_initialization()

st.title('Initialize storages')

check_for_uploaded_prices()


if st.button('Initialize storage'):
    st.session_state.storages.append(initialize_storage(st.session_state.prices))

for storage in st.session_state.storages:
    st.write(f'Gas Storage ID: {storage.id}')
