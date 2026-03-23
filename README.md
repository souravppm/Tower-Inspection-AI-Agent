# 🗼 Tower Inspection AI Agent

An enterprise-grade AI system that analyzes telecom tower structural health using **Drone Imagery**, **Vision AI (YOLOv8)**, and **LLM Agents (Groq Llama 3)**.

## 🚀 Key Features
- **Object Detection:** Custom YOLOv8 model trained to detect tower joints and structural anomalies.
- **Agentic Analysis:** Uses Groq (Llama 3.1) to convert raw detection data into human-readable executive reports.
- **Interactive Dashboard:** Streamlit UI featuring health gauge charts, confidence sliders, and batch processing.
- **State Persistence:** Smart disk-caching to maintain analysis results across page refreshes.
- **Automated Reporting:** Generates downloadable PDF summaries of structural health.

## 📂 Dataset & Model Weights (Important)
To keep the repository clean, the heavy dataset and model weight (`.pt`) files are not included. To run this locally:
1. **Model Weights:** Train the model using `train_model.py` or place your pre-trained custom YOLOv8 `.pt` file in the root directory.
2. **Dataset:** Place your drone imagery dataset inside a `datasets/` folder in the root directory.

## ⚙️ Setup & Installation
1. Clone this repository: `git clone https://github.com/souravppm/Tower-Inspection-AI-Agent.git`
2. Create a virtual environment and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Create a `.env` file in the root directory and add your Groq API key:
   `GROQ_API_KEY=your_api_key_here`
5. Run the application: `python -m streamlit run frontend/app.py`
```"
