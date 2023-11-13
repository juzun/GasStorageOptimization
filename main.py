import streamlit as st
from gas_tank.gs_optim import GasStorage
import gas_tank.utils as utils
import datetime as dt


def main(state=None):

    st.title('Nazdar')

    if 'solve_clicked' not in st.session_state:
        st.session_state.solve_clicked = False

    storage = utils.initialize_storage()
    st.write(f'ID zásobníku: {storage.id}')

    st.button("Reset", type="primary")
    if st.button('Say hello'):
        st.write('Why hello there')
    else:
        st.write('Goodbye')

    def solve_button():
        st.session_state.solve_clicked = True
        storage.solve_model(solver_name='scip', stream_solver=True)
    
    st.button('Solve', on_click = solve_button)

    st.write('Completed')
    st.write(storage.objective)



if __name__ == "__main__":
    main()