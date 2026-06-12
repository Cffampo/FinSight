import streamlit as st
from query_data import query_rag

st.set_page_config(page_title="FinSight", page_icon="📈")

st.title("📈 FinSight")
st.subheader("AI-powered financial document assistant")
st.write("Ask questions about Apple, JPMorgan, or Morningstar SEC filings.")

question = st.text_input("Ask a question about the financial documents:")

if st.button("Ask"):
    if question:
        with st.spinner("Searching documents and generating answer..."):
            result = query_rag(question)
            if result:
                st.markdown("### Answer")
                st.write(result["answer"])
                st.markdown("### Sources")
                for source in set(result["sources"]):
                    st.write(f"📄 {source}")
    else:
        st.warning("Please enter a question.")