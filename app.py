import streamlit as st
import os
import pdfplumber
import docx
import faiss
import numpy as np
import matplotlib.pyplot as plt
import requests
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import tiktoken

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")
client = Groq(api_key=api_key)

# Initialize FAISS and Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  
faiss_index = faiss.IndexFlatL2(dimension)
documents = []
doc_embeddings = []

# Health Risk Keywords
HEALTH_RISK_KEYWORDS = {

    "cancer": 10, "hypertension": 8, "diabetes": 9, 
    "heart disease": 9, "stroke": 8
}

# GDPR Compliance Keywords
GDPR_KEYWORDS = {
    "personal data": 8.0, "data protection": 8.5, "privacy": 7.5, "confidentiality": 8.0,
    "data breach": 10.0, "consent": 7.8, "explicit consent": 8.2, "informed consent": 8.0,
    "data subject": 7.5, "right to access": 7.0, "right to erasure": 7.2, "right to rectification": 7.0,
    "data portability": 7.0, "gdpr": 9.0, "data controller": 8.0, "data processor": 8.0,
    "lawful basis": 7.5, "data retention": 7.5
}

# Initialize LLM for document comparison
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3, api_key=api_key)

# Function to send summary via email
def send_email(user_email, document_summary):
    FORMSPREE_URL = "https://formspree.io/f/mrbpzlor"  # Replace with your Formspree endpoint
    payload = {
        "email": user_email,
        "subject": "Your Legal Document Summary",
        "message": document_summary
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(FORMSPREE_URL, json=payload, headers=headers)
    return "✅ Email sent successfully!" if response.status_code == 200 else f"❌ Failed to send email. Error: {response.text}"

# Extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            text += extracted + "\n" if extracted else ""
    return text.strip()

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Split large text into smaller chunks
def split_text(text, max_tokens=5000):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

# Summarize large document
def summarize_large_document(text):
    text_chunks = split_text(text, max_tokens=5000)
    summaries = []
    for chunk in text_chunks:
        prompt = f"Summarize the following legal document section:\n\n{chunk}"
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  #? REPLACE *************************
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=500
        )
        summaries.append(response.choices[0].message.content.strip())
    return "\n\n".join(summaries)

# Improved Risk Analysis (from friend's code)
def assess_risks(document_text):
    if not document_text or document_text.strip() == "":
        return {"error": "No content to analyze for risks."}
    
    risk_categories = {
    "Chronic Disease Risks": ["diabetes", "hypertension", "heart disease", "cancer"],
    "Acute Condition Risks": ["stroke", "heart attack", "sepsis", "pneumonia"],
    "Lifestyle Risks": ["obesity", "smoking", "alcoholism", "sedentary"],
    "Infectious Disease Risks": ["infection", "tuberculosis", "hepatitis", "flu"],
    "Mental Health Risks": ["depression", "anxiety", "bipolar", "schizophrenia"]
}
    
    text = document_text.lower()
    results = {}
    total_score = 0
    
    for category, keywords in risk_categories.items():
        category_score = 0
        keyword_hits = []
        for keyword in keywords:
            count = text.count(keyword)
            if count > 0:
                keyword_hits.append({"keyword": keyword, "count": count})
                category_score += count
        results[category] = {"score": category_score, "keywords": keyword_hits}
        total_score += category_score
    
    risk_level = "Low" if total_score <= 10 else "Medium" if total_score <= 20 else "High"
    
    visualization_data = {
        "categories": list(risk_categories.keys()),
        "scores": [results[cat]["score"] for cat in risk_categories.keys()],
        "colors": ["green" if s == 0 else "yellow" if s < 3 else "orange" if s < 5 else "red" for s in [results[cat]["score"] for cat in risk_categories.keys()]]
    }
    
    sentences = [s.strip() for s in text.replace(".", ". ").split(". ") if s.strip()]
    risk_sentences = [s for s in sentences if any(keyword in s for cat, keywords in risk_categories.items() for keyword in keywords)]
    
    return {
        "risk_level": risk_level,
        "total_score": total_score,
        "category_results": results,
        "top_risk_sentences": risk_sentences[:5],
        "visualization_data": visualization_data
    }

# Visualize Risks (from friend's code)
def plot_risk_analysis(risk_results, title):
    if "visualization_data" not in risk_results:
        st.write("❌ No visualization data available")
        return
    
    viz_data = risk_results["visualization_data"]
    categories, scores, colors = viz_data["categories"], viz_data["scores"], viz_data["colors"]
    
    if not any(scores):
        st.write("✅ No risks detected!")
        return
    
    # Bar Chart
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.bar(categories, scores, color=colors)
    ax1.set_xlabel("Risk Categories")
    ax1.set_ylabel("Risk Score")
    ax1.set_title(title)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f"{height}", ha="center", va="bottom")
    
    # Pie Chart
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    non_zero_categories = [categories[i] for i, s in enumerate(scores) if s > 0]
    non_zero_scores = [s for s in scores if s > 0]
    
    if non_zero_scores:
        ax2.pie(non_zero_scores, labels=non_zero_categories, autopct="%1.1f%%", startangle=90, colors=["yellow", "orange", "red"][:len(non_zero_scores)])
        ax2.set_title("Risk Distribution")
    else:
        ax2.text(0.5, 0.5, "No risks detected", ha="center", va="center")
    
    plt.tight_layout()
    st.pyplot(fig1)
    st.pyplot(fig2)

# GDPR Compliance Check
def check_gdpr_compliance(text):
    gdpr_issues = {}
    for line in text.split("\n"):
        detected_keywords = [word for word in GDPR_KEYWORDS if word in line.lower()]
        if detected_keywords:
            risk_score = sum(GDPR_KEYWORDS[word] for word in detected_keywords)
            gdpr_issues[", ".join(detected_keywords)] = risk_score
    return gdpr_issues

# Add document to FAISS
def add_to_faiss(text):
    global faiss_index, documents, doc_embeddings
    embedding = embedding_model.encode([text])
    faiss_index.add(np.array(embedding, dtype=np.float32))
    documents.append(text)
    doc_embeddings.append(embedding)

# Legal Chatbot with RAG
def legal_chatbot(user_question, document_text):
    question_embedding = embedding_model.encode([user_question])
    distances, indices = faiss_index.search(np.array(question_embedding, dtype=np.float32), k=1)
    
    if indices[0][0] == -1 or not documents:
        return "I couldn't find relevant information in the document."
    
    relevant_chunk = documents[indices[0][0]]
    prompt = (
        f"You are a legal assistant specialized in compliance and GDPR. Based on the document context:\n\n"
        f"**Document Context:**\n{relevant_chunk}\n\n"
        f"**User Question:**\n{user_question}\n\n"
        f"Provide a concise and accurate response."
    )
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",     #? REPLACE*************************
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=300
    )
    return response.choices[0].message.content.strip()

# Document Comparison (from friend's code)
def compare_documents(doc1_text, doc2_text):
    prompt = PromptTemplate(
        template="Compare these two health documents and highlight key differences in diagnoses, treatments, risk factors, or patient outcomes:\n"
                 "Document 1: {doc1}\nDocument 2: {doc2}\nOrganize by sections or topics.",
        input_variables=["doc1", "doc2"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    comparison_result = chain.run(doc1=doc1_text, doc2=doc2_text)
    return comparison_result

# Streamlit UI
st.set_page_config(page_title="🧠 AI Healthcare Risk Predictor & Report Summarizer", layout="wide")
st.title("🏥 AI-Powered Healthcare Analysis")


# Sidebar for file upload
st.sidebar.header("📂 Upload Your Legal Document")
st.sidebar.text("Note: This AI assistant is for informational purposes only.")
uploaded_file = st.sidebar.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
uploaded_file2 = st.sidebar.file_uploader("Upload a second document for comparison (optional)", type=["pdf", "docx", "txt"])

document_text = None
document_text2 = None

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        document_text = extract_text_from_docx(uploaded_file)
    else:
        document_text = uploaded_file.getvalue().decode("utf-8")
    add_to_faiss(document_text)

if uploaded_file2 is not None:
    file_extension2 = uploaded_file2.name.split(".")[-1]
    if file_extension2 == "pdf":
        document_text2 = extract_text_from_pdf(uploaded_file2)
    elif file_extension2 == "docx":
        document_text2 = extract_text_from_docx(uploaded_file2)
    else:
        document_text2 = uploaded_file2.getvalue().decode("utf-8")

# Tabs for functionalities
if document_text:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📜 Summarization", "⚠ Risk Analysis", "🩺 Health Compliance",
        "📥 Download Reports", "📧 Email Summary", "📑 Document Comparison"
    ])

    with tab1:
        st.subheader("📜 Document Summary")
        summary = summarize_large_document(document_text)
        st.write(summary)
        summary_filename = "Legal_Summary.txt"
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(summary)
        with open(summary_filename, "rb") as f:
            st.download_button("📥 Download Summary (TXT)", f, file_name=summary_filename)

    with tab2:
        st.subheader("⚠ Identified Risks")
        risks = assess_risks(document_text)
        plot_risk_analysis(risks, "⚠ Risk Analysis of Legal Document")

    with tab3:
        st.subheader("🛡 GDPR Compliance Check")
        gdpr_issues = check_gdpr_compliance(document_text)
        if gdpr_issues:
            fig, ax = plt.subplots(figsize=(8, 5))
            clauses = list(gdpr_issues.keys())
            scores = list(gdpr_issues.values())
            ax.barh(clauses, scores, color="red")
            ax.set_xlabel("Risk Score")
            ax.set_title("🛡 GDPR Compliance Risks")
            st.pyplot(fig)
        else:
            st.success("✅ No GDPR compliance issues detected!")
        
        st.subheader("💬 Health Assistance Chatbot")
        user_question = st.text_input("Ask a question about the document:")
        if user_question:
            chatbot_response = legal_chatbot(user_question, document_text)
            st.write("🩺 **Response:**")
            st.write(chatbot_response)

    with tab4:
        st.subheader("📥 Download Reports")
        gdpr_report = "Health Compliance Report:\n\n" + "\n".join([f"{k}: {v}" for k, v in gdpr_issues.items()])
        gdpr_filename = "Health_Compliance_Report.txt"
        with open(gdpr_filename, "w", encoding="utf-8") as f:
            f.write(gdpr_report)
        with open(gdpr_filename, "rb") as f:
            st.download_button("📥 Download Health Compliance Report (TXT)", f, file_name=gdpr_filename)

    with tab5:
        st.subheader("📧 Receive Summary via Email")
        user_email = st.text_input("Enter your email:")
        if st.button("Send Summary"):
            if user_email:
                email_status = send_email(user_email, summary)
                st.success(email_status)
            else:
                st.warning("⚠ Please enter a valid email address.")

    with tab6:
        st.subheader("📑 Document Comparison")
        if document_text2:
            st.write("Comparing the two uploaded documents...")
            comparison_result = compare_documents(document_text, document_text2)
            st.markdown(comparison_result)
            comparison_filename = "document_comparison.txt"
            with open(comparison_filename, "w", encoding="utf-8") as f:
                f.write(comparison_result)
            with open(comparison_filename, "rb") as f:
                st.download_button("📥 Download Comparison (TXT)", f, file_name=comparison_filename)
        else:
            st.warning("Please upload a second document to enable comparison.")