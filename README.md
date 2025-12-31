# Intelligent-Resume-Based-Job-Suggestion

This project is an end-to-end AI-powered job recommendation platform that intelligently analyzes resumes, extracts skills, retrieves real-time job openings, identifies skill gaps, and generates personalized career insights using a fully automated pipeline.

# 🚀 Key Features

✔ Smart Resume Upload & Secure Processing

Resume upload via Streamlit UI

Files stored securely in AWS S3 with metadata

Lambda-based virus scan & validation

S3 event trigger automatically starts processing workflow

# ✔ AI-Powered Resume Understanding

AWS Textract / Comprehend for structured text extraction

Amazon Titan Embeddings for generating resume vectors

OpenAI models used for:

Skill extraction & summarization

Education & work experience parsing

Achievement interpretation

Processed profile stored in MongoDB Atlas with embeddings

# ✔ RAG-Based Job Retrieval Layer

Fetches live job listings from Adzuna, JSearch, Indeed APIs

Embeds job descriptions & performs semantic similarity search

LLM (Claude/OpenAI) enriches each job with:

Summary

Required skills

Responsibilities

# ✔ Interactive Streamlit Dashboard

Displays Top 20 job recommendations

Skill-gap heatmaps & match explanations

Course recommendations via Coursera API

Daily auto-refresh via API Gateway → Lambda

User feedback loop continuously improves ranking accuracy

# 📊 Outcomes & Impact

Highly contextual job matches using hybrid semantic ranking

Automatically generated personalized summaries

End-to-end automated pipeline from S3 → Lambda → MongoDB → Streamlit

Analytics including:

Average match score

Skill-gap distribution

Industry/role fit insights# Intelligent-Resume-Based-Job-Suggestion
