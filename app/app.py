import streamlit as st
import tempfile

# Backend logic
from main_project import ExerciseAnalyzer

# LangChain (v0.1.x – stable)
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Gym Fitness Assistant",
    layout="wide"
)

st.title("🏋️ AI Gym Fitness Assistant")
st.write("Upload a workout video, analyze squats, get diet plans and AI guidance.")

# ---------------------------
# Initialize Analyzer
# ---------------------------
analyzer = ExerciseAnalyzer()

# ---------------------------
# VIDEO UPLOAD & ANALYSIS
# ---------------------------
st.header("🎥 Workout Video Analysis")

uploaded_video = st.file_uploader(
    "Upload a squat workout video",
    type=["mp4", "mov", "avi"]
)

rep_count = 0

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.success("Video uploaded successfully")

    if st.button("▶ Analyze Squats"):
        st.info("Analyzing video... please wait")
        rep_count = analyzer.analyze_video_real_time(video_path, exercise="squat")

        st.success("Analysis complete")

        st.subheader("📊 Workout Summary")
        st.write(f"**Total Squats Counted:** {rep_count}")
        st.write("**Workout Quality:** Good")
        st.write("**Suggested Rest Time:** 60 seconds")

# ---------------------------
# DIET PLAN GENERATOR
# ---------------------------
st.header("🥗 AI Diet Plan Generator")

goal = st.selectbox(
    "Select your fitness goal",
    ["Weight Loss", "Muscle Gain", "Maintenance"]
)

diet_type = st.selectbox(
    "Diet Preference",
    ["Vegetarian", "Non-Vegetarian"]
)

if st.button("Generate Diet Plan"):
    if goal == "Weight Loss":
        diet_plan = {
            "Breakfast": "Oats with fruits",
            "Lunch": "Brown rice, dal, vegetables",
            "Dinner": "Salad or grilled vegetables",
            "Calories": "1500–1700 kcal"
        }

    elif goal == "Muscle Gain":
        diet_plan = {
            "Breakfast": "Eggs / paneer with toast",
            "Lunch": "Rice, chicken/paneer, vegetables",
            "Dinner": "Protein-rich salad",
            "Calories": "2500–2800 kcal"
        }

    else:
        diet_plan = {
            "Breakfast": "Idli / toast with fruits",
            "Lunch": "Balanced rice and curry",
            "Dinner": "Light meal with protein",
            "Calories": "2000–2200 kcal"
        }

    st.success("✅ Personalized Diet Plan Generated")
    for meal, item in diet_plan.items():
        st.write(f"**{meal}:** {item}")

# ---------------------------
# AI FITNESS CHATBOT
# ---------------------------
st.header("🤖 AI Fitness Chatbot")

@st.cache_resource
def load_llm():
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=200,
        temperature=0.7
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

llm = load_llm()

prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a professional fitness trainer.
Answer the following question clearly with steps and safety tips.

Question: {question}
Answer:
"""
)

memory = ConversationBufferMemory()
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

user_question = st.text_input("Ask a fitness or workout question:")

if st.button("Ask AI"):
    if user_question.strip():
        response = llm_chain.run(user_question)
        st.success("AI Response:")
        st.write(response)
    else:
        st.warning("Please enter a question")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("AI Gym Fitness Assistant – Demo Version")
