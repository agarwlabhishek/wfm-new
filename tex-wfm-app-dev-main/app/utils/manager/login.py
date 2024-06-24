import os
import yaml
import logging
import streamlit as st
from typing import Dict


# Create a logger
logger = logging.getLogger(__name__)


def load_allowed_users() -> Dict[str, str]:
    """
    Load allowed usernames and passwords from the YAML file.
    """
    try:
        # Construct the path to the file in the same directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "credentials.yaml")

        with open(file_path, 'r') as file:
            allowed_users = yaml.safe_load(file)
        return allowed_users

    except Exception as e:
        logger.error(f"Error while loading allowed users: {e}")
        raise e

        
def is_valid_user(username: str, password: str, allowed_users: Dict[str, str]) -> bool:
    """
    Check if the provided username and password are valid.
    """
    try:
        if username in allowed_users:
            return allowed_users[username] == password
        return False

    except Exception as e:
        logger.error(f"Error while validating user: {e}")
        raise e

        
def login_page() -> None:
    """
    Displays login page and processes the login.
    """
    try:
        st.header('Login')

        username = st.text_input('Username')
        password = st.text_input('Password', type='password')

        if st.button('Login'):
            allowed_users = load_allowed_users()
            if is_valid_user(username, password, allowed_users):
                st.success('Logged in successfully!')
                st.session_state.logged_in = True
            else:
                st.error('Invalid username or password', icon="🚨")

    except Exception as e:
        logger.error(f"Error in login page: {e}")
        raise e

        
__all__ = ['login_page']
