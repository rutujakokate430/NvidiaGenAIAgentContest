Human Resource Portal- HR GPT 🗒️
The Human Resource Portal is a comprehensive tool designed for HR executives to streamline candidate screening, perform background verification, identify better fitting job roles, and determine average salary ranges. This project leverages NVIDIA's NIM endpoints for inference and NeMo™ Guardrails to ensure accurate and compliant responses from language models.

Features
Candidate Screening: Utilizes RAG to extract tokens from resumes and job descriptions, creating embeddings for matching skills.

Background Verification: A LangChain agent performs web searches to verify candidate backgrounds, calculating estimated ages based on available data.

Better Fitting Job Roles: LangChain agents search for job openings that closely match candidates' skills from their resumes.

Average Salary Range: LangChain agents search for and provide the average salary range for specific job roles based on market standards.

Setup
Requirements
Ensure you have Python 3.7+ installed.

Installation
Clone the repository:

bash
Copy code
git clone <repository-url>
cd human-resource-portal
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Configuration
Set environment variables:

bash
Copy code
export SERPAPI_API_KEY=<your-serpapi-api-key>
export NVIDIA_API_KEY=<your-nvidia-api-key>
Configure Guardrails:

Ensure your Guardrails configuration files (config.yml, prompts.yml) are correctly set up for security and compliance.
Usage
Run the Streamlit application:

bash
Copy code
streamlit run app.py
Access the Human Resource Portal via your web browser.

Example Workflow
Select the experience level and enter the job description in the sidebar.
Upload the candidate's resume.
Explore various sections:
Resume Screening: See the percentage of skills matched with the job description.
Candidate Background: View estimated candidate age and background verification details.
Other Better Fitting Roles: Discover hyperlinked URLs to job roles better suited to the candidate.
Salary Range: Find the average salary range for the specified job description.
Credits
NVIDIA NIM Endpoints: For providing powerful inference capabilities.
NVIDIA NeMo™ Guardrails: Ensuring secure and compliant responses from language models.
