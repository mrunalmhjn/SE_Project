# import streamlit as st
# import tempfile
# from backend.analyzer import analyze_video

# st.set_page_config(page_title="🎬 Video Analyzer", layout="wide")

# st.title("🎬 Video Sentiment & Virality Analyzer")
# st.markdown("AI-powered multimodal analysis (Visual + Audio + NLP)")

# uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# if uploaded_file:

#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         tmp.write(uploaded_file.read())
#         video_path = tmp.name

#     st.video(video_path)

#     if st.button("🚀 Analyze Video"):

#         with st.spinner("Analyzing..."):
#             result = analyze_video(video_path)

#         st.success("Analysis Complete!")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.metric("🎭 Sentiment",
#                       result["final"]["sentiment_label"],
#                       result["final"]["sentiment_score"])

#         with col2:
#             st.metric("🔥 Virality",
#                       result["final"]["virality_label"],
#                       result["final"]["virality_score"])

#         st.subheader("🧠 Insights")

#         st.write("**Transcript Preview:**")
#         st.info(result["text"]["text"])

#         st.write("**Visual Analysis:**")
#         st.json(result["visual"])

#         st.write("**Text Analysis:**")
#         st.json(result["text"])


import streamlit as st
import tempfile
from backend.core import run_full_analysis

st.set_page_config(layout="wide")

st.title("🎬 Video Sentiment & Virality Analyzer")

file = st.file_uploader("Upload Video")

if file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        path = tmp.name

    st.video(path)

    if st.button("Analyze"):

        with st.spinner("Processing..."):
            result = run_full_analysis(path)

        st.success("Done!")

        col1, col2 = st.columns(2)

        col1.metric("Sentiment", result["sentiment"])
        col2.metric("Virality", result["virality"])

        st.image(result["dashboard"])
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv("outputs/frame_data.csv")

        st.subheader("📈 Frame-Level Analysis")

        # Sentiment plot
        fig1, ax = plt.subplots()
        ax.plot(df["sentiment"])
        ax.set_title("Sentiment Over Time")
        st.pyplot(fig1)

        # Virality plot
        fig2, ax = plt.subplots()
        ax.plot(df["virality"])
        ax.set_title("Virality Over Time")
        st.pyplot(fig2)

        # st.download_button("Download Summary", open("outputs/summary.csv","rb"))
        # st.download_button("Download Frame Data", open("outputs/frame_data.csv","rb"))
        with open("outputs/summary.csv", "rb") as f:
            st.download_button(
                label="📄 Download Summary CSV",
                data=f,
                file_name="video_analysis_summary.csv",
                mime="text/csv"
            )

        with open("outputs/frame_data.csv", "rb") as f:
            st.download_button(
                label="📈 Download Frame Data CSV",
                data=f,
                file_name="frame_level_results.csv",
                mime="text/csv"
            )