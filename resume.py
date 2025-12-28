# resume_app.py - Main Streamlit Application with Lambda Integration
import os
import uuid
import json
import re
import tempfile
import base64
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymongo
import boto3
import requests
import pdfplumber
import docx
from dotenv import load_dotenv
from difflib import SequenceMatcher
from botocore.exceptions import ClientError

# =============================
# LOAD ENVIRONMENT VARIABLES
# =============================
load_dotenv()

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "resume-bucket")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# HuggingFace (Optional)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Lambda Configuration
LAMBDA_FUNCTION_NAME = os.getenv("LAMBDA_FUNCTION_NAME", "ResumeProcessorLambda")

# =============================
# ADD LAMBDA FUNCTION HERE
# =============================
def call_lambda_function(file_content, file_name):
    """Call AWS Lambda function to process resume"""
    try:
        # Check if AWS credentials are available
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            st.warning("⚠️ AWS credentials not found. Using local processing instead.")
            return None
        
        # Initialize Lambda client
        lambda_client = boto3.client(
            'lambda',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        # Prepare payload
        payload = {
            "body": json.dumps({
                "file_name": file_name,
                "file_content": base64.b64encode(file_content).decode('utf-8'),
                "content_type": "application/pdf" if file_name.lower().endswith('.pdf') else
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if file_name.lower().endswith('.docx') else
                               "text/plain",
                "resume_id": str(uuid.uuid4())
            })
        }
        
        # Invoke Lambda with progress indicator
        with st.spinner("🤖 Processing with AWS Lambda..."):
            response = lambda_client.invoke(
                FunctionName=LAMBDA_FUNCTION_NAME,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
        
        # Parse response
        response_payload = json.loads(response['Payload'].read())
        
        if response['StatusCode'] == 200:
            result = json.loads(response_payload['body'])
            st.success("✅ Lambda processing complete!")
            return result
        else:
            st.error(f"Lambda invocation failed: {response_payload}")
            return None
            
    except Exception as e:
        st.error(f"Failed to call Lambda: {e}")
        return None

# =============================
# INITIALIZE CLIENTS
# =============================
def initialize_clients():
    """Initialize AWS and MongoDB clients"""
    clients = {}
    
    # AWS S3 Client
    try:
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            s3_client = boto3.client(
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
            # Test S3 connection
            s3_client.list_buckets()
            clients['s3'] = s3_client
            st.sidebar.success("✅ Connected to AWS S3")
            
            # Also initialize Lambda client if S3 works
            lambda_client = boto3.client(
                'lambda',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
            clients['lambda'] = lambda_client
        else:
            st.sidebar.warning("⚠️ AWS credentials not found")
            clients['s3'] = None
            clients['lambda'] = None
    except Exception as e:
        st.sidebar.error(f"❌ AWS connection failed: {e}")
        clients['s3'] = None
        clients['lambda'] = None
    
    # MongoDB Client
    try:
        mongo_client = pymongo.MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000
        )
        mongo_client.server_info()  # Test connection
        db = mongo_client.job_db
        clients['mongo'] = db
        st.sidebar.success("✅ Connected to MongoDB")
    except Exception as e:
        st.sidebar.error(f"❌ MongoDB connection failed: {e}")
        # Create in-memory database as fallback
        clients['mongo'] = None
        clients['in_memory'] = {
            'resumes': [],
            'matches': [],
            'jobs': []
        }
    
    return clients

# =============================
# RESUME PROCESSING FUNCTIONS
# =============================
def extract_text_from_file(file_path, file_extension):
    """Extract text from PDF, DOCX, or TXT files"""
    text = ""
    
    try:
        if file_extension.lower() == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"
        
        elif file_extension.lower() == 'docx':
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        
        elif file_extension.lower() == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        return text.strip()
    
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def generate_embedding(text):
    """Generate text embedding using HuggingFace or local method"""
    if HUGGINGFACE_API_KEY:
        try:
            url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
            response = requests.post(
                url,
                headers=headers,
                json={"inputs": text[:1000]},  # Limit text for API
                timeout=30
            )
            if response.status_code == 200:
                return response.json()[0]
        except Exception as e:
            st.warning(f"HuggingFace embedding failed: {e}")
    
    # Fallback: Simple TF-IDF like embedding
    words = text.lower().split()
    unique_words = list(set(words))
    embedding = [words.count(word) / len(words) for word in unique_words]
    # Pad or truncate to 384 dimensions (same as all-MiniLM-L6-v2)
    if len(embedding) > 384:
        embedding = embedding[:384]
    else:
        embedding += [0] * (384 - len(embedding))
    return embedding

def extract_skills_from_text(text):
    """Extract skills from resume text"""
    # Comprehensive skill database
    skill_database = {
        "Programming": ["Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin"],
        "Web Development": ["HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask", "FastAPI"],
        "Data Science": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy"],
        "Databases": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "Oracle", "SQLite", "DynamoDB"],
        "Cloud & DevOps": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "CI/CD", "Jenkins", "Git", "Linux"],
        "Data Tools": ["Power BI", "Tableau", "Excel", "Apache Spark", "Hadoop", "Kafka", "Airflow"],
        "Soft Skills": ["Leadership", "Communication", "Teamwork", "Problem Solving", "Project Management", "Agile"]
    }
    
    found_skills = []
    text_lower = text.lower()
    
    # Check each skill category
    for category, skills in skill_database.items():
        for skill in skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
    
    # Pattern-based extraction
    patterns = [
        r'(?:Skills|Technologies|Expertise)[:\s]+([^\.]+)',
        r'Proficient in[:\s]+([^\.]+)',
        r'Experience with[:\s]+([^\.]+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Split by common delimiters
            parts = re.split(r'[,;/&•\-]', match)
            for part in parts:
                part = part.strip()
                if part and len(part) > 2:
                    # Capitalize and add if not already in list
                    capitalized = part.title()
                    if capitalized not in found_skills:
                        found_skills.append(capitalized)
    
    return list(set(found_skills))  # Remove duplicates

def analyze_resume_sections(text):
    """Analyze and summarize resume sections"""
    sections = {
        'summary': '',
        'experience': '',
        'education': '',
        'skills': '',
        'projects': ''
    }
    
    lines = text.split('\n')
    current_section = 'other'
    
    for line in lines:
        line_lower = line.strip().lower()
        
        # Identify section headers
        if any(keyword in line_lower for keyword in ['summary', 'objective', 'profile']):
            current_section = 'summary'
        elif any(keyword in line_lower for keyword in ['experience', 'employment', 'work history']):
            current_section = 'experience'
        elif any(keyword in line_lower for keyword in ['education', 'academic', 'qualification']):
            current_section = 'education'
        elif any(keyword in line_lower for keyword in ['skills', 'technical skills', 'competencies']):
            current_section = 'skills'
        elif any(keyword in line_lower for keyword in ['projects', 'portfolio', 'achievements']):
            current_section = 'projects'
        elif line.strip():
            # Add content to current section
            if current_section in sections:
                sections[current_section] += line + " "
    
    # Create overall summary
    summary_parts = []
    for section, content in sections.items():
        if content.strip():
            summary_parts.append(f"{section.title()}: {content[:100]}...")
    
    return " | ".join(summary_parts) if summary_parts else "No sections identified"

# =============================
# STORAGE FUNCTIONS
# =============================
def upload_to_s3(file_content, file_name, s3_client, bucket_name):
    """Upload file to S3 bucket"""
    if s3_client is None:
        return None
    
    try:
        # Generate unique key
        file_extension = file_name.split('.')[-1].lower()
        s3_key = f"resumes/{uuid.uuid4()}.{file_extension}"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=file_content,
            ContentType='application/pdf' if file_extension == 'pdf' else 
                       'application/vnd.openxmlformats-officedocument.wordprocessingml.document' if file_extension == 'docx' else
                       'text/plain'
        )
        
        # Generate public URL (or pre-signed URL for private buckets)
        s3_url = f"https://{bucket_name}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        
        return {
            's3_key': s3_key,
            's3_url': s3_url,
            'bucket': bucket_name
        }
    
    except Exception as e:
        st.error(f"Failed to upload to S3: {e}")
        return None

def store_in_mongodb(resume_data, clients):
    """Store resume data in MongoDB"""
    try:
        if clients.get('mongo') is not None:
            # Store in resumes collection
            result = clients['mongo'].resumes.insert_one(resume_data)
            return result.inserted_id
        else:
            # Store in memory
            clients['in_memory']['resumes'].append(resume_data)
            return "memory_id_" + str(len(clients['in_memory']['resumes']))
    except Exception as e:
        st.error(f"Failed to store in MongoDB: {e}")
        return None

def store_matches_in_db(matches, clients):
    """Store match results in database"""
    try:
        if clients.get('mongo') is not None:
            # Clear previous matches
            clients['mongo'].matches.delete_many({})
            if matches:
                clients['mongo'].matches.insert_many(matches)
        else:
            clients['in_memory']['matches'] = matches
        return True
    except Exception as e:
        st.error(f"Failed to store matches: {e}")
        return False

# =============================
# JOB MATCHING FUNCTIONS
# =============================
def get_job_listings():
    """Get job listings from APIs or mock data"""
    # Try to fetch from Adzuna API
    ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
    ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
    
    if ADZUNA_APP_ID and ADZUNA_APP_KEY:
        try:
            url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
            params = {
                "app_id": ADZUNA_APP_ID,
                "app_key": ADZUNA_APP_KEY,
                "what": "software engineer",
                "results_per_page": 10
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                jobs = response.json().get("results", [])
                processed_jobs = []
                for job in jobs:
                    processed_jobs.append({
                        "id": job.get("id", str(uuid.uuid4())),
                        "title": job.get("title", "Unknown"),
                        "company": job.get("company", {}).get("display_name", "Unknown"),
                        "description": job.get("description", ""),
                        "location": job.get("location", {}).get("display_name", "Remote"),
                        "salary": job.get("salary_min"),
                        "url": job.get("redirect_url"),
                        "posted_date": job.get("created", datetime.now().isoformat())
                    })
                return processed_jobs
        except Exception as e:
            st.warning(f"Adzuna API failed: {e}")
    
    # Fallback to mock data
    return get_mock_jobs()

def get_mock_jobs():
    """Return mock job data for testing"""
    mock_jobs = [
        {
            "id": "1",
            "title": "Senior Python Developer",
            "company": "Tech Innovations Inc.",
            "description": "We're looking for a Senior Python Developer with 5+ years of experience in building scalable web applications. Must have expertise in FastAPI, Django, and AWS services.",
            "required_skills": ["Python", "FastAPI", "Django", "AWS", "SQL", "Docker"],
            "location": "Remote",
            "salary": 120000,
            "posted_date": "2024-01-15"
        },
        {
            "id": "2",
            "title": "Machine Learning Engineer",
            "company": "AI Solutions Corp",
            "description": "Join our AI team to develop cutting-edge machine learning models. Experience with PyTorch, TensorFlow, and computer vision required.",
            "required_skills": ["Python", "Machine Learning", "PyTorch", "TensorFlow", "Computer Vision", "AWS"],
            "location": "San Francisco, CA",
            "salary": 150000,
            "posted_date": "2024-01-14"
        },
        {
            "id": "3",
            "title": "Full Stack Developer",
            "company": "WebTech Solutions",
            "description": "Full Stack Developer needed for React and Node.js applications. Experience with MongoDB and cloud deployment required.",
            "required_skills": ["JavaScript", "React", "Node.js", "MongoDB", "AWS", "Docker"],
            "location": "New York, NY",
            "salary": 110000,
            "posted_date": "2024-01-13"
        },
        {
            "id": "4",
            "title": "Data Scientist",
            "company": "Data Insights LLC",
            "description": "Data Scientist with expertise in NLP and predictive modeling. Must know Python, scikit-learn, and have experience with large datasets.",
            "required_skills": ["Python", "Machine Learning", "NLP", "Pandas", "Scikit-learn", "SQL"],
            "location": "Remote",
            "salary": 130000,
            "posted_date": "2024-01-12"
        },
        {
            "id": "5",
            "title": "DevOps Engineer",
            "company": "Cloud Systems Inc.",
            "description": "DevOps Engineer with Docker, Kubernetes, and CI/CD experience. AWS certification preferred.",
            "required_skills": ["Docker", "Kubernetes", "AWS", "CI/CD", "Terraform", "Linux"],
            "location": "Austin, TX",
            "salary": 125000,
            "posted_date": "2024-01-11"
        }
    ]
    return mock_jobs

def calculate_match_score(resume_text, resume_skills, job):
    """Calculate match score between resume and job"""
    # 1. Semantic similarity
    sem_similarity = SequenceMatcher(
        None,
        resume_text.lower(),
        job.get("description", "").lower()
    ).ratio()
    
    # 2. Skill overlap
    job_skills = job.get("required_skills", [])
    skill_overlap = 0
    if resume_skills and job_skills:
        matched_skills = set(resume_skills) & set(job_skills)
        skill_overlap = len(matched_skills) / max(len(job_skills), 1)
    
    # 3. Composite score
    final_score = (0.5 * sem_similarity) + (0.5 * skill_overlap)
    
    # 4. Missing skills
    missing_skills = list(set(job_skills) - set(resume_skills))
    
    return {
        'semantic_score': round(sem_similarity, 3),
        'skill_match': round(skill_overlap, 3),
        'final_score': round(final_score, 3),
        'missing_skills': missing_skills[:5]
    }

def match_resume_to_jobs(resume_text, resume_skills, jobs):
    """Match resume against job listings"""
    matches = []
    
    for job in jobs:
        scores = calculate_match_score(resume_text, resume_skills, job)
        
        match_data = {
            "match_id": str(uuid.uuid4()),
            "job_id": job["id"],
            "job_title": job["title"],
            "company": job["company"],
            "description": job["description"][:200] + "...",
            "location": job.get("location", "N/A"),
            "salary": job.get("salary"),
            "semantic_score": scores['semantic_score'],
            "skill_match": scores['skill_match'],
            "final_score": scores['final_score'],
            "missing_skills": scores['missing_skills'],
            "matched_at": datetime.now().isoformat()
        }
        matches.append(match_data)
    
    # Sort by final score
    matches.sort(key=lambda x: x["final_score"], reverse=True)
    return matches[:10]

# =============================
# VISUALIZATION FUNCTIONS
# =============================
def create_skill_gap_visualization(matches):
    """Create visualization for missing skills"""
    if not matches:
        return None
    
    # Collect missing skills data
    missing_skills = {}
    for match in matches:
        for skill in match.get('missing_skills', []):
            missing_skills[skill] = missing_skills.get(skill, 0) + 1
    
    if not missing_skills:
        return None
    
    # Prepare data
    df = pd.DataFrame(list(missing_skills.items()), columns=['Skill', 'Count'])
    df = df.sort_values('Count', ascending=True).tail(10)  # Top 10 missing skills
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df['Skill'], df['Count'], color='lightcoral')
    
    # Customize
    ax.set_xlabel('Frequency in Job Matches')
    ax.set_title('Top Missing Skills in Job Recommendations')
    ax.invert_yaxis()  # Highest on top
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', va='center')
    
    plt.tight_layout()
    return fig

def create_score_distribution_chart(matches):
    """Create histogram of match scores"""
    if not matches:
        return None
    
    scores = [match['final_score'] for match in matches]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Match Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Job Match Scores')
    
    # Add average line
    avg_score = sum(scores) / len(scores)
    ax.axvline(avg_score, color='red', linestyle='--', 
               label=f'Average: {avg_score:.2f}')
    ax.legend()
    
    plt.tight_layout()
    return fig

# =============================
# STREAMLIT UI - COMPLETED
# =============================
def main():
    st.set_page_config(
        page_title="SmartHire AI - Resume Analyzer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Initialize clients
        if 'clients' not in st.session_state:
            st.session_state.clients = initialize_clients()
        
        st.markdown("---")
        st.subheader("Storage Status")
        
        # FIXED: Compare with None instead of boolean testing
        s3_status = "✅ Connected" if st.session_state.clients.get('s3') is not None else "❌ Not Connected"
        mongo_status = "✅ Connected" if st.session_state.clients.get('mongo') is not None else "❌ Not Connected"
        lambda_status = "✅ Available" if st.session_state.clients.get('lambda') is not None else "❌ Not Available"
        
        st.write(f"AWS S3: {s3_status}")
        st.write(f"MongoDB: {mongo_status}")
        st.write(f"AWS Lambda: {lambda_status}")
        
        st.markdown("---")
        st.subheader("Options")
        
        # ADDED: Processing mode selection
        processing_mode = st.radio(
            "Processing Mode",
            ["Local Processing", "AWS Lambda Processing"],
            index=0,
            help="Local: Fast, no cloud. Lambda: Scalable, uses AWS services"
        )
        
        use_live_jobs = st.checkbox("Use Live Job Data", value=False)
        show_advanced = st.checkbox("Show Advanced Options", value=False)
        
        if show_advanced:
            semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.5)
            skill_weight = st.slider("Skill Weight", 0.0, 1.0, 0.5)
        
        if st.button("Clear All Data", type="secondary"):
            # FIXED: Check if mongo exists and is not None
            if st.session_state.clients.get('mongo') is not None:
                st.session_state.clients['mongo'].resumes.delete_many({})
                st.session_state.clients['mongo'].matches.delete_many({})
            st.success("All data cleared!")
    
    # Main content
    st.title("🤖 SmartHire AI - Intelligent Resume Analyzer")
    st.markdown("Upload your resume to get AI-powered job recommendations and skill analysis")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload Resume", "🔍 Analysis Results", "📊 Analytics", "💾 Storage Info"])
    
    with tab1:
        st.header("Upload Your Resume")
        
        uploaded_file = st.file_uploader(
            "Choose a resume file",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT (Max 10MB)"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_size = uploaded_file.size / (1024 * 1024)  # MB
            if file_size > 10:
                st.error("File too large! Please upload a file smaller than 10MB")
                return
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{file_size:.2f} MB")
            with col3:
                st.metric("File Type", uploaded_file.type)
            
            # ADDED: Show processing mode
            st.info(f"**Processing Mode:** {processing_mode}")
            
            # Process resume
            if st.button("🚀 Analyze Resume", type="primary"):
                with st.spinner("Processing your resume..."):
                    
                    if processing_mode == "AWS Lambda Processing":
                        # =============================================
                        # LAMBDA PROCESSING PATH
                        # =============================================
                        lambda_result = call_lambda_function(
                            uploaded_file.getvalue(),
                            uploaded_file.name
                        )
                        
                        if lambda_result and lambda_result.get("status") == "success":
                            # Extract data from Lambda response
                            resume_id = lambda_result.get("resume_id", str(uuid.uuid4()))
                            skills = lambda_result.get("skills", [])
                            summary = lambda_result.get("summary", "")
                            matches = lambda_result.get("matches", [])
                            s3_info = lambda_result.get("s3_info", {})
                            mongo_response = lambda_result.get("mongo_response", {})
                            
                            # Display Lambda results
                            st.success(f"✅ Lambda processing complete! Found {len(matches)} job matches.")
                            
                            # Show extracted information
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("🛠️ Extracted Skills (from Lambda)")
                                if skills:
                                    for skill in skills[:15]:
                                        st.markdown(f"• {skill}")
                                    if len(skills) > 15:
                                        st.caption(f"... and {len(skills) - 15} more skills")
                                else:
                                    st.info("No skills detected in the resume")
                            
                            with col2:
                                st.subheader("📋 Resume Summary (from Lambda)")
                                st.info(summary)
                            
                            # Store in session state
                            st.session_state['resume_data'] = {
                                "resume_id": resume_id,
                                "file_name": uploaded_file.name,
                                "skills": skills,
                                "summary": summary,
                                "s3_info": s3_info,
                                "mongo_response": mongo_response,
                                "processed_by": "AWS Lambda"
                            }
                            st.session_state['matches'] = matches
                            
                            # Store in local MongoDB if available (for consistency)
                            if st.session_state.clients.get('mongo') is not None:
                                resume_doc = {
                                    "resume_id": resume_id,
                                    "file_name": uploaded_file.name,
                                    "file_type": uploaded_file.type,
                                    "file_size": file_size,
                                    "upload_date": datetime.now().isoformat(),
                                    "skills": skills,
                                    "summary": summary,
                                    "s3_info": s3_info,
                                    "processed_by": "AWS Lambda",
                                    "lambda_response": mongo_response
                                }
                                store_in_mongodb(resume_doc, st.session_state.clients)
                                store_matches_in_db(matches, st.session_state.clients)
                            
                            st.balloons()
                            
                        else:
                            st.warning("⚠️ Lambda processing failed. Falling back to local processing...")
                            # Fall back to local processing
                            processing_mode = "Local Processing"
                    
                    if processing_mode == "Local Processing":
                        # =============================================
                        # LOCAL PROCESSING PATH
                        # =============================================
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, 
                                                        suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Extract text
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        resume_text = extract_text_from_file(tmp_path, file_extension)
                        
                        if not resume_text:
                            st.error("Could not extract text from the file. Please try another file.")
                            return
                        
                        # Show extracted text
                        with st.expander("📝 View Extracted Text"):
                            st.text_area("Resume Content", resume_text[:1500], height=200)
                        
                        # Extract skills and analyze
                        resume_skills = extract_skills_from_text(resume_text)
                        resume_summary = analyze_resume_sections(resume_text)
                        
                        # Display extracted information
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🛠️ Extracted Skills")
                            if resume_skills:
                                for skill in resume_skills[:15]:  # Show first 15 skills
                                    st.markdown(f"• {skill}")
                                if len(resume_skills) > 15:
                                    st.caption(f"... and {len(resume_skills) - 15} more skills")
                            else:
                                st.info("No skills detected in the resume")
                        
                        with col2:
                            st.subheader("📋 Resume Summary")
                            st.info(resume_summary)
                        
                        # Store in S3
                        s3_info = None
                        if st.session_state.clients.get('s3') is not None:
                            s3_info = upload_to_s3(
                                uploaded_file.getvalue(),
                                uploaded_file.name,
                                st.session_state.clients['s3'],
                                BUCKET_NAME
                            )
                        
                        # Prepare resume data for MongoDB
                        resume_data = {
                            "resume_id": str(uuid.uuid4()),
                            "file_name": uploaded_file.name,
                            "file_type": uploaded_file.type,
                            "file_size": file_size,
                            "upload_date": datetime.now().isoformat(),
                            "resume_text": resume_text[:5000],  # Store first 5000 chars
                            "skills": resume_skills,
                            "summary": resume_summary,
                            "s3_info": s3_info,
                            "processed_by": "Local Processing"
                        }
                        
                        # Store in MongoDB
                        db_id = store_in_mongodb(resume_data, st.session_state.clients)
                        
                        if db_id:
                            st.success(f"✅ Resume stored successfully (ID: {db_id})")
                            if s3_info:
                                st.info(f"📁 S3 Location: {s3_info['s3_key']}")
                        
                        # Get job listings and match
                        jobs = get_job_listings() if use_live_jobs else get_mock_jobs()
                        matches = match_resume_to_jobs(resume_text, resume_skills, jobs)
                        
                        # Store matches
                        store_matches_in_db(matches, st.session_state.clients)
                        
                        # Store in session state
                        st.session_state['resume_data'] = resume_data
                        st.session_state['matches'] = matches
                        st.session_state['jobs'] = jobs
                        
                        st.balloons()
                        st.success(f"✅ Local processing complete! Found {len(matches)} job matches.")
    
    with tab2:
        st.header("Job Recommendation Results")
        
        if 'matches' in st.session_state and st.session_state['matches']:
            matches = st.session_state['matches']
            
            # Show processing source
            resume_data = st.session_state.get('resume_data', {})
            processed_by = resume_data.get('processed_by', 'Local Processing')
            st.caption(f"Processed by: {processed_by}")
            
            # Sort options
            col1, col2 = st.columns([1, 3])
            with col1:
                sort_option = st.selectbox(
                    "Sort by",
                    ["Best Match", "Skill Match", "Semantic Match", "Company"],
                    key="sort_select"
                )
            
            # Sort matches
            if sort_option == "Best Match":
                matches.sort(key=lambda x: x['final_score'], reverse=True)
            elif sort_option == "Skill Match":
                matches.sort(key=lambda x: x['skill_match'], reverse=True)
            elif sort_option == "Semantic Match":
                matches.sort(key=lambda x: x['semantic_score'], reverse=True)
            elif sort_option == "Company":
                matches.sort(key=lambda x: x['company'])
            
            # Display matches
            for i, match in enumerate(matches):
                with st.expander(f"#{i+1}: {match['job_title']} at {match['company']} | Score: {match['final_score']:.3f}", 
                               expanded=(i < 3)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**🏢 Company:** {match['company']}")
                        st.markdown(f"**📍 Location:** {match['location']}")
                        if match.get('salary'):
                            st.markdown(f"**💰 Salary:** ${match['salary']:,}")
                        st.markdown(f"**📝 Description:** {match['description']}")
                    
                    with col2:
                        # Score cards
                        st.metric("Overall Score", f"{match['final_score']:.3f}")
                        st.metric("Skill Match", f"{match['skill_match']:.3f}")
                        st.metric("Semantic Match", f"{match['semantic_score']:.3f}")
                        
                        # Missing skills
                        if match['missing_skills']:
                            st.warning("**Missing Skills:**")
                            for skill in match['missing_skills']:
                                st.markdown(f"• {skill}")
                        else:
                            st.success("**Perfect Skill Match!**")
                    
                    st.markdown("---")
        else:
            st.info("👈 Upload and analyze a resume first to see job recommendations")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if 'matches' in st.session_state and st.session_state['matches']:
            matches = st.session_state['matches']
            resume_data = st.session_state.get('resume_data', {})
            
            # Show processing source
            processed_by = resume_data.get('processed_by', 'Local Processing')
            st.caption(f"Processed by: {processed_by}")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_score = sum(m['final_score'] for m in matches) / len(matches)
                st.metric("Average Match Score", f"{avg_score:.3f}")
            with col2:
                st.metric("Total Matches", len(matches))
            with col3:
                top_score = max(m['final_score'] for m in matches)
                st.metric("Top Match Score", f"{top_score:.3f}")
            with col4:
                skill_count = len(resume_data.get('skills', []))
                st.metric("Skills Found", skill_count)
            
            st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_score_distribution_chart(matches)
                if fig1:
                    st.pyplot(fig1)
                else:
                    st.info("No data for score distribution")
            
            with col2:
                fig2 = create_skill_gap_visualization(matches)
                if fig2:
                    st.pyplot(fig2)
                else:
                    st.info("No missing skills data")
            
            # Skill Analysis
            st.markdown("---")
            st.subheader("Skill Analysis")
            if resume_data.get('skills'):
                skills_df = pd.DataFrame({
                    'Skill': resume_data['skills'],
                    'Category': ['Technical'] * len(resume_data['skills'])  # Simplified
                })
                
                # Show skills in a table
                st.dataframe(skills_df, use_container_width=True)
            else:
                st.info("No skills data available")
                
        else:
            st.info("👈 Upload and analyze a resume first to see analytics")
    
    with tab4:
        st.header("Storage Information")
        
        if 'clients' in st.session_state:
            clients = st.session_state.clients
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Database Status")
                if clients.get('mongo') is not None:
                    try:
                        # Get counts from MongoDB
                        resume_count = clients['mongo'].resumes.count_documents({})
                        match_count = clients['mongo'].matches.count_documents({})
                        
                        st.metric("Resumes Stored", resume_count)
                        st.metric("Job Matches Stored", match_count)
                        
                        # Show recent resumes
                        if resume_count > 0:
                            recent_resumes = list(clients['mongo'].resumes.find(
                                {}, 
                                {'file_name': 1, 'upload_date': 1, 'skills_count': {'$size': '$skills'}}
                            ).sort('upload_date', -1).limit(5))
                            
                            st.subheader("Recent Resumes")
                            for resume in recent_resumes:
                                st.write(f"📄 {resume.get('file_name', 'Unknown')} - {resume.get('upload_date', 'Unknown')}")
                    except Exception as e:
                        st.error(f"Error accessing MongoDB: {e}")
                else:
                    st.warning("MongoDB not connected")
                    # Show in-memory data
                    if clients.get('in_memory'):
                        st.metric("Resumes in Memory", len(clients['in_memory'].get('resumes', [])))
                        st.metric("Matches in Memory", len(clients['in_memory'].get('matches', [])))
            
            with col2:
                st.subheader("AWS S3 Status")
                if clients.get('s3') is not None:
                    try:
                        # List S3 bucket contents
                        response = clients['s3'].list_objects_v2(Bucket=BUCKET_NAME, Prefix="resumes/")
                        
                        if 'Contents' in response:
                            s3_count = len(response['Contents'])
                            total_size = sum(obj['Size'] for obj in response['Contents']) / (1024*1024)
                            
                            st.metric("Files in S3", s3_count)
                            st.metric("Total Storage", f"{total_size:.2f} MB")
                            
                            if s3_count > 0:
                                st.subheader("Recent S3 Files")
                                for obj in response['Contents'][:5]:
                                    st.write(f"📁 {obj['Key'].split('/')[-1]} - {obj['Size']/1024:.1f} KB")
                        else:
                            st.info("No files in S3 bucket")
                    except Exception as e:
                        st.error(f"Error accessing S3: {e}")
                else:
                    st.warning("S3 not connected")
        
        # Session State Info
        st.markdown("---")
        st.subheader("Session State")
        if st.button("Show Session State", type="secondary"):
            session_keys = list(st.session_state.keys())
            st.write(f"Session keys: {session_keys}")
            
            if 'resume_data' in st.session_state:
                st.json(st.session_state['resume_data'], expanded=False)
            
            if 'matches' in st.session_state:
                st.write(f"Number of matches: {len(st.session_state['matches'])}")

# =============================
# RUN THE APPLICATION
# =============================
if __name__ == "__main__":
    main()