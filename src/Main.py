from utils import (
    check_session_initialization,
    reset_session_state,
    authentication_process,
)
import streamlit as st


def main():
    st.set_page_config(layout="wide", page_title="Home")

    authentication_process()
    check_session_initialization()

    st.header("Manual")
    st.write(
        """
        This is an application for gas storage optimization.

        To start, first import prices on Prices tab.

        Then initialize storages you want to optimize on tab Storages. You can choose from given templates.

        These templates can be edited in the source directory of the code, in 'src/data/storages.json'.

        Each storage can be then optimized on Optimize tab, or you can run optimization for all of the storages.

        After successful optimization you can find table and graph export of each storage on Export and Graph tabs. You can also download these exports and see total export of all storages.
        """
    )

    st.button(
        "Restart session",
        on_click=reset_session_state,
        help="Click this button to restart all variables.",
    )


if __name__ == "__main__":
    main()
