import os
import re
import plotly.graph_objects as go
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import initialize_agent, AgentType
from langchain.agents import AgentExecutor
from langchain.tools import Tool
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from key import nvidia_api_key, serp_api_key

# Set API keys
os.environ["SERPAPI_API_KEY"] = serp_api_key
os.environ['NVIDIA_API_KEY'] = nvidia_api_key

# Initialize NVIDIA model
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Load guardrails configuration
config = RailsConfig.from_path("./config")
guardrails = RunnableRails(config)

# Streamlit setup
st.set_page_config(page_title="HR GPT Portal", layout="wide")

# CSS Styling
st.markdown("""
    <style>
        body { background-color: #000000; color: #76B900; }
        .stApp { background-color: #000000; color: #76B900; }
        h1, h2, h3, h4, h5, h6 { color: #76B900; }
        .css-1d391kg { color: #76B900; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div style="text-align: center; color: #76B900; font-size: 32px;">💬 HR GPT: AI Hiring Assistant 🗒️</div>', unsafe_allow_html=True)

# Sidebar inputs
experience = st.sidebar.selectbox("Experience Level", options=["Entry-level", "Mid-Level", "Senior-Level"])
jd = st.sidebar.text_area("Enter the job description:", max_chars=None)
pdf = st.sidebar.file_uploader("Upload the candidate resume", type="pdf")

# Extract text from PDF
def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = "".join([page.extract_text() for page in pdf_reader.pages])
    return text

# Extract match percentage from response
def extract_percentage(text):
    match = re.search(r'(\d+)%', text)
    return int(match.group(1)) if match else None

# Generate donut chart for match percentage
def generate_donut_chart(percentage):
    fig = go.Figure(data=[go.Pie(labels=['Match', 'Mismatch'], values=[percentage, 100 - percentage], hole=.6)])
    fig.update_layout(showlegend=False, annotations=[dict(text=f'{percentage}%', x=0.5, y=0.5, font_size=20, showarrow=False)])
    return fig

# Extract candidate name from resume
def extract_candidate_name(text, knowledge_base):
    query = "Extract candidate name from the resume."
    docs = knowledge_base.similarity_search(query)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    name_match = re.search(r'Candidate Name:\s*(.*)', response)
    return name_match.group(1).strip() if name_match else "Unknown Candidate"

# Define agent functions
def analyze_resume(resume_text, experience, jd):
    """Extracts skills match percentage and missing skills."""
    prompt = PromptTemplate(
        input_variables=["experience", "jd"],
        template=(
            "Analyze this resume and compare it with the job description: {jd} for experience level {experience}. "
            "Provide the percentage match (in bold) and list missing skills."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.invoke({"experience": experience, "jd": jd})["output"]

def fetch_candidate_background(candidate_name):
    """Fetches candidate background using LinkedIn search."""
    search_query = f"Search LinkedIn for {candidate_name}'s background (brief, max 70 words)."
    tools = initialize_agent(["serpapi"], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return tools.run(search_query)

def find_better_jobs(candidate_name, missing_skills):
    """Finds better fitting job roles based on missing skills."""
    search_query = f"List job roles that better fit {candidate_name} given missing skills: {missing_skills}."
    tools = initialize_agent(["serpapi"], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return tools.run(search_query)

def estimate_salary(jd):
    """Finds salary range for the job description."""
    search_query = f"Find the market salary range for: {jd}."
    tools = initialize_agent(["serpapi"], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return tools.run(search_query)

# Define tools for the agent pipeline
resume_analysis_tool = Tool(
    name="Resume Analysis",
    func=lambda inputs: analyze_resume(inputs["resume_text"], inputs["experience"], inputs["jd"]),
    description="Analyzes a resume and compares it with a job description to determine match percentage."
)

background_check_tool = Tool(
    name="Background Check",
    func=lambda inputs: fetch_candidate_background(inputs["candidate_name"]),
    description="Fetches candidate background details from LinkedIn."
)

job_search_tool = Tool(
    name="Job Search",
    func=lambda inputs: find_better_jobs(inputs["candidate_name"], inputs["missing_skills"]),
    description="Finds job roles that better fit the candidate based on missing skills."
)

salary_estimation_tool = Tool(
    name="Salary Estimation",
    func=lambda inputs: estimate_salary(inputs["jd"]),
    description="Estimates salary range based on the job description."
)

# Agent chaining execution
agent_executor = AgentExecutor(
    tools=[resume_analysis_tool, background_check_tool, job_search_tool, salary_estimation_tool],
    verbose=True
)

if pdf is not None:
    # Extract text from PDF
    resume_text = extract_text_from_pdf(pdf)

    # Process resume text into vector embeddings
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(resume_text)
    embeddings = NVIDIAEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Extract candidate name
    candidate_name = extract_candidate_name(resume_text, knowledge_base)

    # Run chained agent pipeline
    final_output = agent_executor.run({
        "resume_text": resume_text,
        "experience": experience,
        "jd": jd,
        "candidate_name": candidate_name
    })

    # Extract and display results
    match_percentage = extract_percentage(final_output["Resume Analysis"])
    missing_skills = final_output["Resume Analysis"].split("Missing Skills:")[-1].strip()

    st.subheader("Resume Screening Results")
    st.write(final_output["Resume Analysis"])
    if match_percentage:
        st.plotly_chart(generate_donut_chart(match_percentage))

    st.subheader("Candidate Background")
    st.write(final_output["Background Check"])

    st.subheader("Better Fitting Job Roles")
    st.write(final_output["Job Search"])

    st.subheader("Estimated Salary Range")
    st.write(final_output["Salary Estimation"])
