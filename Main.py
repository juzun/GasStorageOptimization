from gas_tank.utils import *


def main(state=None):

    st.button('Start session', on_click=reset_session_state, help='Click this button to initialize or restart all needed variables.')
    
    if 'session_initialized' in st.session_state:
        if st.session_state.session_initialized:
            st.info('There is a session running.')


if __name__ == "__main__":
    main()