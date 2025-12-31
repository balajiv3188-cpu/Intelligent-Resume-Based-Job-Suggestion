import json
import boto3
import PyPDF2
import io
import re
from datetime import datetime
from botocore.exceptions import ClientError

# Initialize S3 client for reading PDFs from S3 (if needed)
s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda function to process PDF resumes and extract key information
    """
    try:
        print(f"Received event: {json.dumps(event)}")
        
        # Parse input - can accept different input formats
        if 'body' in event and isinstance(event['body'], str):
            # If body is stringified JSON (API Gateway proxy)
            body = json.loads(event['body'])
        else:
            # Direct invocation or other formats
            body = event
        
        # Extract PDF content - can come from different sources
        pdf_content = None
        
        # Option 1: PDF content directly in the event (base64 encoded)
        if 'pdf_content' in body and body['pdf_content']:
            import base64
            pdf_content = base64.b64decode(body['pdf_content'])
        
        # Option 2: S3 bucket and key provided
        elif 's3_bucket' in body and 's3_key' in body:
            try:
                s3_response = s3_client.get_object(
                    Bucket=body['s3_bucket'],
                    Key=body['s3_key']
                )
                pdf_content = s3_response['Body'].read()
            except ClientError as e:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': f'S3 error: {str(e)}',
                        'processed': False
                    })
                }
        
        # Option 3: Raw text already extracted
        elif 'text_content' in body:
            return process_text_content(body['text_content'])
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'No PDF content, S3 reference, or text content provided',
                    'processed': False
                })
            }
        
        # Process PDF if we have content
        if pdf_content:
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_content)
            
            if text:
                # Parse resume information
                resume_info = parse_resume_info(text)
                resume_info['processing_timestamp'] = datetime.now().isoformat()
                resume_info['lambda_request_id'] = context.aws_request_id
                resume_info['processed'] = True
                
                return {
                    'statusCode': 200,
                    'body': json.dumps(resume_info, default=str)
                }
            else:
                return {
                    'statusCode': 400,
                    'body': json.dumps({
                        'error': 'Failed to extract text from PDF',
                        'processed': False
                    })
                }
        
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}',
                'processed': False
            })
        }

def extract_text_from_pdf(pdf_content):
    """
    Extract text content from PDF bytes
    """
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += f"--- Page {page_num} ---\n"
                    text += page_text + "\n\n"
            except Exception as page_error:
                print(f"Error extracting text from page {page_num}: {str(page_error)}")
                continue
        
        return text.strip() if text else None
    
    except Exception as e:
        print(f"Error in extract_text_from_pdf: {str(e)}")
        return None

def parse_resume_info(text):
    """
    Parse extracted text to find resume information with enhanced extraction
    """
    info = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "experience_summary": None,
        "education_summary": None,
        "total_experience": None,
        "job_titles": [],
        "companies": [],
        "universities": [],
        "raw_text_preview": text[:1000] + "..." if len(text) > 1000 else text,
        "word_count": len(text.split()),
        "page_count": len(re.findall(r'--- Page \d+ ---', text)) or 1
    }
    
    # Clean and normalize text
    text_lower = text.lower()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # 1. Extract Email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text, re.IGNORECASE)
    if emails:
        info["email"] = emails[0]
        # Remove email from text for cleaner parsing
        text = re.sub(email_pattern, '', text, flags=re.IGNORECASE)
    
    # 2. Extract Phone Numbers (multiple formats)
    phone_patterns = [
        r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US/Canada
        r'\d{10}',  # Simple 10 digit
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'  # 3-3-4 format
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        if phones:
            info["phone"] = phones[0]
            break
    
    # 3. Extract Name (usually in first few lines, title case, 2-3 words)
    for i, line in enumerate(lines[:10]):
        line_clean = re.sub(r'[^a-zA-Z\s-]', '', line).strip()
        words = line_clean.split()
        
        # Heuristics for name detection
        if 2 <= len(words) <= 4:
            # Check if looks like a name (title case, not all caps, not too long)
            if (line_clean.istitle() or 
                (line_clean[0].isupper() and not line_clean.isupper())):
                
                # Avoid common resume section headers
                blacklist = ['summary', 'experience', 'education', 'skills', 
                           'objective', 'projects', 'certifications', 'references']
                if not any(blackword in line_lower for blackword in blacklist):
                    info["name"] = line_clean
                    break
    
    # 4. Extract Skills (comprehensive list)
    skill_categories = {
        'Programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift', 'kotlin'],
        'Web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'express'],
        'Cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins'],
        'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'dynamodb'],
        'Data Science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'r'],
        'Tools': ['git', 'jira', 'confluence', 'slack', 'visual studio', 'intellij', 'eclipse']
    }
    
    found_skills = []
    for category, skills in skill_categories.items():
        for skill in skills:
            skill_pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(skill_pattern, text_lower, re.IGNORECASE):
                # Capitalize skill name for display
                display_skill = skill.title() if len(skill.split()) == 1 else skill
                found_skills.append({
                    'skill': display_skill,
                    'category': category
                })
    
    # Deduplicate skills
    unique_skills = []
    seen_skills = set()
    for skill in found_skills:
        if skill['skill'].lower() not in seen_skills:
            unique_skills.append(skill)
            seen_skills.add(skill['skill'].lower())
    
    info["skills"] = unique_skills
    
    # 5. Extract Experience Summary
    exp_keywords = ['experience', 'work history', 'employment', 'professional experience']
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in exp_keywords):
            # Capture next few lines as experience context
            exp_context = ' '.join(lines[i+1:i+5])
            if exp_context:
                info["experience_summary"] = exp_context[:500] + "..."
            break
    
    # 6. Extract Education Summary
    edu_keywords = ['education', 'university', 'college', 'degree', 'bachelor', 'master', 'phd']
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in edu_keywords):
            edu_context = ' '.join(lines[i+1:i+5])
            if edu_context:
                info["education_summary"] = edu_context[:500] + "..."
            break
    
    # 7. Extract Job Titles (common patterns)
    title_patterns = [
        r'\b(?:Senior|Junior|Lead|Principal)?\s*(?:Software|Data|DevOps|ML|AI|Frontend|Backend|Full Stack)?\s*Engineer\b',
        r'\b(?:Data|Business|Systems|Product)?\s*Analyst\b',
        r'\b(?:Project|Product|Program)\s*Manager\b',
        r'\b(?:Software|Web|Application)\s*Developer\b',
        r'\bArchitect\b',
        r'\bScientist\b',
        r'\bConsultant\b',
        r'\bSpecialist\b'
    ]
    
    for pattern in title_patterns:
        titles = re.findall(pattern, text, re.IGNORECASE)
        for title in titles:
            if title.lower() not in [t.lower() for t in info["job_titles"]]:
                info["job_titles"].append(title.strip())
    
    # 8. Extract Companies (look for capitalized multi-word entities)
    company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
    potential_companies = re.findall(company_pattern, text)
    
    # Filter out common words that aren't companies
    non_companies = {'United States', 'Summary', 'Experience', 'Education', 'Skills'}
    for company in potential_companies:
        if (len(company.split()) >= 2 and 
            company not in non_companies and
            company.lower() not in text_lower[:200]):  # Avoid section headers
            info["companies"].append(company)
    
    # 9. Extract Universities
    edu_keywords_extended = ['university', 'college', 'institute', 'school']
    for line in lines:
        for keyword in edu_keywords_extended:
            if keyword in line.lower():
                # Clean the line to extract university name
                clean_line = re.sub(r'[^a-zA-Z\s]', '', line)
                words = clean_line.split()
                if len(words) >= 2:
                    potential_uni = ' '.join(words[:3])
                    info["universities"].append(potential_uni)
                break
    
    # 10. Try to extract total experience
    exp_years_pattern = r'(\d+)\+?\s*(?:years?|yrs?)'
    exp_matches = re.findall(exp_years_pattern, text_lower)
    if exp_matches:
        try:
            years = [int(y) for y in exp_matches]
            info["total_experience"] = f"{max(years)}+ years"
        except:
            pass
    
    return info

def process_text_content(text):
    """
    Process already extracted text content
    """
    try:
        resume_info = parse_resume_info(text)
        resume_info['processing_timestamp'] = datetime.now().isoformat()
        resume_info['processed'] = True
        
        return {
            'statusCode': 200,
            'body': json.dumps(resume_info, default=str)
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': f'Error processing text: {str(e)}',
                'processed': False
            })
        }