import os
import re
import plotly.graph_objects as go
from key import nvidia_api_key, serp_api_key
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

# Set environment variables
os.environ["SERPAPI_API_KEY"] = serp_api_key
os.environ['NVIDIA_API_KEY'] = nvidia_api_key

# Initialize NVIDIA model
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Load guardrails configuration
config = RailsConfig.from_path("./config")
guardrails = RunnableRails(config)

# Sample Chain with Guardrails
prompt = PromptTemplate(
    input_variables=["experience", "jd"],
    template=("You are performing candidate screening. "
              "With the uploaded resume of the candidate, give the exact percentage of Skills matched to the job description: {jd} and experience level {experience}. "
              "Do not list matching skills, only show skills match percentage in bold, like 'Match Score:'. "
              "Also, list the skills that are missing in the Resume to match the Job description. "
              "Do not describe in detail, only give keywords.")
)

chain = LLMChain(llm=llm, prompt=prompt)
chain_with_guardrails = guardrails | chain

# Streamlit configuration
st.set_page_config(page_title="Human Resource Portal", layout="wide")

# CSS Styling
st.markdown("""
    <style>
        /* General settings */
        body {
            background-color: #000000;
            color: #76B900;
        }
        .stApp {
            background-color: #000000;
            color: #76B900;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #76B900;
        }
        .css-1d391kg {
            color: #76B900;
        }
        .css-1cpxqw2 a {
            color: #00FF00;
        }
        /* File uploader settings */
        .css-1nnc9jn {
            background-color: #333333;
            color: #FFFFFF;
        }
        /* Donut chart settings */
        .css-18e3th9 {
            margin-top: -50px;
        }
        /* Section settings */
        .section {
            background-color: #000000;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div style="text-align: center; color: #76B900; font-size: 32px;">💬 Human Resource Portal 🗒️</div>', unsafe_allow_html=True)

# Sidebar inputs
experience = st.sidebar.selectbox("Experience Level", options=["Entry-level", "Mid-Level", "Senior-Level"])
jd = st.sidebar.text_area("Enter the job description:", max_chars=None)
pdf = st.sidebar.file_uploader("Upload the candidate resume", type="pdf")

# Function to extract text from PDF and process match
def generate_match(experience, jd):
    response = chain_with_guardrails.invoke({"experience": experience, "jd": jd})
    return response['output']

def generate_donut_chart(percentage):
    fig = go.Figure(data=[go.Pie(labels=['Match', 'Mismatch'],
                                 values=[percentage, 100 - percentage],
                                 hole=.6)])
    fig.update_layout(showlegend=False,
                      annotations=[dict(text=f'{percentage}%', x=0.5, y=0.5, font_size=20, showarrow=False)],
                      width=330, height=330)  # Increase width and height by 30%
    return fig

def extract_percentage(text):
    match = re.search(r'(\d+)%', text)
    if match:
        return int(match.group(1))
    return None

def extract_candidate_name(text, knowledge_base):
    user_question = "Based on all the information from the Candidate resume, Only print- Candidate Name: Name"
    docs = knowledge_base.similarity_search(user_question)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=user_question)
    name_match = re.search(r'Candidate Name:\s*(.*)', response)
    if name_match:
        return name_match.group(1).strip()
    return "Unknown Candidate"

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    embeddings = NVIDIAEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

# Display the dashboard
col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume Screening")
    if experience and jd and pdf:
        result = generate_match(experience, jd)
        st.write(result)
        match_percentage = extract_percentage(result)
        if match_percentage is not None:
            st.plotly_chart(generate_donut_chart(match_percentage), use_container_width=False, height=150)

with col2:
    st.subheader("Candidate Background")
    if pdf is not None:
        candidate_name = extract_candidate_name(text, knowledge_base)
        user_question = f"Based on all the information from the Candidate resume, what can be a rough estimate of Candidate Age. Only print- Candidate Name: {candidate_name} , Age: Age"
        docs = knowledge_base.similarity_search(user_question)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        st.write(response)
        tools = load_tools(["serpapi"])
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)
        search_query = f"use the Hyperlinked LinkedIn URL from the resume of the Candidate:{candidate_name}, and tell me background of this candidate in brief (in 70 words only) that you found from your good search for this specific candidate. Tell me things you which are not present in the resume. Use information which is available to the public"
        search_response = agent.run(search_query)
        st.write("Candidate Background")
        st.write(search_response)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Other Better Fitting Roles")
    if pdf is not None:
        tools = load_tools(["serpapi"])
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)
        search_query = f"give Only Hyperlinked URL's to those Job Roles that are a better fit for the candidate: {candidate_name}"
        search_response = agent.run(search_query)
        st.write("Better Fit Job Roles:")
        st.write(search_response)

with col4:
    st.subheader("Salary Range")
    if pdf is not None:
        tools = load_tools(["serpapi"])
        agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)
        search_query = f"Based on the Job description: {jd}, give the Average Salary Range according to the market Salary for this specific role, Average Salary Range:"
        search_response = agent.run(search_query)
        st.write("Average Salary Range:")
        st.write(search_response)
