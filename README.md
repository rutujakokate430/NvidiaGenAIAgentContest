# HR GPT - A Comprehensive Tool for HR Executives

Welcome to **HR GPT**, a one-stop tool designed for HR executives to streamline the hiring process. This tool offers capabilities for candidate screening, background verification, finding better fitting job roles, and checking average salary ranges according to market standards.

<img width="929" alt="image" src="https://github.com/user-attachments/assets/f990307c-44d7-4321-a5cd-718f6eeeabfd" />

<img width="929" alt="image" src="https://github.com/user-attachments/assets/dd1950f1-6e5b-4ffa-a9ec-3f3e42cfff6d" />

## Features

1. **Candidate Screening**:
    - Extracts tokens from resumes and job descriptions using Retrieval-Augmented Generation (RAG).
    - Creates embeddings with NVIDIAEmbeddings.
    - Stores embeddings using FAISS for efficient similarity search.

2. **Background Verification**:
    - Uses a LangChain Agent to perform web searches for background verification.
    - Calculates the candidate's age based on available information.

3. **Job Fit Analysis**:
    - Performs web searches to find job roles better suited to the candidate's skills.
    - Can be tailored to search for job openings in specific companies.

4. **Salary Range Check**:
    - Searches the web for average salary ranges for the candidate's skills and job role.

## Technologies Used

- **NVIDIA NIM Endpoints** for high-performance inference.
- **NVIDIA NeMo™ Guardrails** to control LLM output.
- **Streamlit** for an interactive web interface.
- **Plotly** for data visualization.
- **FAISS** for fast similarity search.
- **LangChain** for building LLM applications.
- **PyPDF2** for PDF handling.

## Installation

1. Clone the repository:
    ```bash
    git clone - https://github.com/rutujakokate430/NvidiaGenAIAgentContest/tree/main
    cd hr-gpt
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your API keys in the `key.py` file:
    ```python
    nvidia_api_key = "your_nvidia_api_key"
    serp_api_key = "your_serp_api_key"
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

## Configuration

- The guardrails configuration is located in the `./config` directory.
- Update the job description and experience level via the Streamlit sidebar.
- Incase of any issues, use **dash.py** with **key.py** for running the application without NVIDIA NeMo™ Guardrails. 
- **key.py** stores the api keys which are then called in **dash.py**

## Usage

1. **Upload the candidate's resume** in PDF format.
2. **Enter the job description** in the provided text area.
3. **Select the experience level** from the dropdown menu.
4. **View the results** for candidate screening, background verification, better fitting roles, and salary range.

## Acknowledgments

- Thanks to **NVIDIA** for the contest and the high-performance **NVIDIA NIM Endpoints** for inference.
- Special appreciation for **NVIDIA NeMo™ Guardrails** for ensuring accurate and controlled LLM outputs.


