import streamlit as st

st.set_page_config(layout="wide")

if "authentication_status" not in st.session_state:
    hide_sidebar_css = """
    <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        button[title="Toggle sidebar"] {
            display: none !important;
        }
    </style>
    """
    st.markdown(hide_sidebar_css, unsafe_allow_html=True)


def login():
    st.session_state["authentication_status"] = True

def logout():
    st.session_state["authentication_status"] = None

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

if st.session_state["authentication_status"] is None:

    st.title("Prueba Tecnica")
    if st.button("Log in"):
        login()
        st.rerun()

if st.session_state["authentication_status"]:


    menu_page = st.Page("pages_app/0_menu.py", title="Menú", icon=":material/menu:")
    part_1 = st.Page("pages_app/1_part_1.py", title="Parte 1", icon=":material/app_registration:")
    
    part_2_scenario_1 = st.Page("pages_app/2_part_2_scenario_1.py", title="Parte 2 - 1", icon=":material/app_registration:")
    
    part_3 = st.Page("pages_app/4_part_3.py", title="Parte 3", icon=":material/app_registration:")

    pg = st.navigation(
        {
            "Menú": [menu_page],
            "Parte_1": [part_1],
            "Parte_2": [part_2_scenario_1],
            "Parte_3": [part_3]

        }
    )

    pg.run()

    if st.button("Log out"):
        logout()
        st.rerun()