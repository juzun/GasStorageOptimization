from gas_tank.utils import *

check_session_initialization()

st.title('Optimize')

check_for_uploaded_prices()
check_for_initialized_storages()

if st.button('Solve'):
    solve_button(st.session_state.storages[0])

if st.session_state.solved:
    for storage in st.session_state.storages:
        st.write(storage.objective)