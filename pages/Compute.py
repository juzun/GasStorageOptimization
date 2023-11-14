import streamlit as st
import gas_tank.utils as utils

if 'prices' in st.session_state:
    prices = st.session_state['prices']
    st.write(prices)

if 'solved' not in st.session_state:
    st.session_state.solved = False

storage = utils.initialize_storage()
st.write(f'ID zásobníku: {storage.id}')

def solve_button():
    storage.solve_model(solver_name='cplex', stream_solver=True)
    st.session_state.solved = True

if st.button('Solve'):
    solve_button()
    st.session_state.solved = True

if st.session_state.solved:
    st.write(storage.objective)
    print(storage.objective)