import streamlit as st
from msal_streamlit_authentication import msal_authentication
import config
import logging

# type: ignore
import requests


def streamlit_authenticate(allowed_ids: list[str]) -> None:
    if config.CLIENT_ID is None:
        raise RuntimeError("Please set CLIENT_ID env. variable.")
    if config.TENANT_ID is None:
        raise RuntimeError("Please set TENANT_ID env. variable.")
    if config.LOGIN_REDIRECT_BASE_URL is None:
        raise RuntimeError("Please set LOGIN_REDIRECT_BASE_URL env. variable.")
    if config.LOGOUT_REDIRECT_BASE_URL is None:
        raise RuntimeError("Please set LOGOUT_REDIRECT_BASE_URL env. variable.")
    login_token = msal_authentication(
        auth={
            "clientId": config.CLIENT_ID,
            "authority": "https://login.microsoftonline.com/" + config.TENANT_ID,
            "redirectUri": (config.LOGIN_REDIRECT_BASE_URL.rstrip("/") + "/"),
            "postLogoutRedirectUri": (
                config.LOGOUT_REDIRECT_BASE_URL.rstrip("/") + "/"
            ),
        },
        cache={
            "cacheLocation": "sessionStorage",
            "storeAuthStateInCookie": False,
        },
        key=1,
    )

    st.session_state.login_token = login_token
    st.session_state.authenticated = False

    if login_token:
        try:
            graph_data = requests.post(
                "https://graph.microsoft.com/v1.0/me/getMemberGroups",
                headers={"Authorization": "Bearer " + login_token["accessToken"]},
                json={"securityEnabledOnly": False},
            )
            graph_data.raise_for_status()
            user_groups = graph_data.json()["value"]
            if all(
                allowed_group_id not in user_groups for allowed_group_id in allowed_ids
            ):
                logging.info(
                    f"ACCESS DENIED: {login_token['account']['username']}."
                    f" Allowed groups: {allowed_ids}."
                    f" User groups: {user_groups}."
                )
                st.error(
                    f"{login_token['account']['username']} does not have access to this application."
                )
                st.stop()
            else:
                st.info(
                    f'You are logged in as {st.session_state.login_token["account"]["username"]}'
                )
                st.session_state.authenticated = True
        except Exception as err:
            st.error(f"Error while fetching user data: {type(err)}: {err}.")
            st.stop()
    else:
        st.stop()
