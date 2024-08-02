import streamlit as st

pages = {
    "Otimização": [
        st.Page("optimize.py", title="Otimize a mistura"),
    ],
    "Documentação": [
        st.Page("data-docs.py", title="Padrão dos dados"),
    ],
}

pg = st.navigation(pages)
pg.run()
