import streamlit as st, requests

st.set_page_config(page_title="EdgeLine Live", layout="wide")
st.title("EdgeLine ✅ — Live Connection Check")

API_BASE = st.secrets.get("API_BASE", "")
API_KEY = st.secrets.get("API_KEY", "")

if not API_BASE:
    st.info("API_BASE not set — running in demo mode.")
else:
    try:
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        r = requests.get(f"{API_BASE}/health", headers=headers, timeout=8)
        if r.ok:
            st.success(f"✅ Connected to API: {API_BASE}")
            st.json(r.json())
        else:
            st.error(f"❌ API error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Connection failed: {e}")
