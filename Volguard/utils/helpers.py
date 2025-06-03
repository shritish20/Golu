import streamlit as st

def find_option_by_strike(option_chain, strike, option_type):
    try:
        for opt in option_chain:
            if opt["strike_price"] == strike:
                if option_type == "CE" and "call_options" in opt:
                    return opt["call_options"]
                if option_type == "PE" and "put_options" in opt:
                    return opt["put_options"]
        st.warning(f":warning: No {option_type} option found for strike {strike}")
        return None
    except Exception as e:
        st.warning(f":warning: Error in find_option_by_strike: {e}")
        return None
