from utils import *


def main():

    st.set_page_config(layout="centered")
    check_session_initialization()

    st.title('Main page')

    st.button('Restart session', on_click=reset_session_state, help='Click this button to restart all variables.')
    
    if 'session_initialized' in st.session_state:
        if st.session_state.session_initialized:
            st.info('There is a session running.')

if __name__ == "__main__":
    main()