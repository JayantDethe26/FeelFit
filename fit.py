import fitz  # PyMuPDF
import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# --------------------------
# User Session Management
# --------------------------
class UserProfileManager:
    """Manages user profiles and information across sessions"""
    def __init__(self):
        self.profiles = {}
    
    def get_profile(self, session_id):
        """Get or create a user profile"""
        if session_id not in self.profiles:
            self.profiles[session_id] = {
                "name": None,
                "age": None,
                "weight": None,
                "height": None,
                "goal": None
            }
        return self.profiles[session_id]
    
    def update_profile(self, session_id, key, value):
        """Update a specific field in the user profile"""
        profile = self.get_profile(session_id)
        profile[key] = value
        return profile

# Global profile manager
profile_manager = UserProfileManager()

# --------------------------
# PDF Processing Section
# --------------------------
def process_pdfs():
    pdf_folder = "pdfs"  
    output_text_file = "datatext.txt"

    if os.path.exists(output_text_file):
        print(f"Using existing '{output_text_file}'")
        return

    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"PDF folder '{pdf_folder}' not found!")

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{pdf_folder}'")

    with open(output_text_file, "w", encoding="utf-8") as txt_file:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Processing: {pdf_file}")
            
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                text = doc[page_num].get_text()
                txt_file.write(text + "\n" + "=" * 80 + "\n")
            print(f"✅ Extracted text from {pdf_file}")

# Initialize PDF processing
try:
    process_pdfs()
except Exception as e:
    print(f"PDF Processing Error: {str(e)}")
    exit(1)

# --------------------------
# LLM Initialization Section
# --------------------------
try:
    # Set up embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # Load and split documents
    loader = TextLoader("datatext.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Initialize Chroma vector store
    persist_directory = "chroma_db"
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("Loaded existing vector store")
    else:
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        print("Created new vector store")

    retriever = vectorstore.as_retriever()

    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="Llama3-8b-8192",
        temperature=0.3,
        max_tokens=1024
    )

except Exception as e:
    print(f"Initialization Error: {str(e)}")
    exit(1)

# --------------------------
# Prompt Template Section
# --------------------------
system_prompt = """You are FeelFit - an expert AI fitness coach specializing in personalized workout plans and nutrition guidance. Your domain expertise includes:
- Weight training program design
- Nutritional planning for specific body types
- Injury prevention and recovery strategies
- Progressive overload techniques
- Macronutrient calculations

User Profile:
Name: {name}
Age: {age}
Weight: {weight}kg
Height: {height}cm
Goal: {goal}

Guidelines:
1. Always ask clarifying questions before making recommendations.
2. Provide exercise alternatives for different equipment availability.
3. Include detailed form instructions for complex lifts.
4. Offer nutrition plans with 3 options for each meal.
5. Use metric units exclusively.
6. Remember the user's name and other personal information when they share it.
7. If personal information is missing, politely ask for it when relevant to fitness advice.
8. Be conversational and friendly while maintaining expertise.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
    ("system", "Context: {context}")
])

# Chain Initialization
llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    input_key="input_documents"
)

# --------------------------
# Information Extraction Functions
# --------------------------
def extract_name(text: str) -> Optional[str]:
    """Extract name from introduction text."""
    patterns = [
        r"(?:my name is|i am|i'm|this is) ([a-z]+(?:\s[a-z]+)*)",  # Match "My name is John Smith"
        r"^([a-z]+(?:\s[a-z]+)*) here",  # Match "John Smith here"
    ]
    
    text_lower = text.lower().strip()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            # Capitalize the first letter of each word in the name
            name = ' '.join(word.capitalize() for word in match.group(1).split())
            return name
    return None

def extract_user_info(text, session_id):
    """Extract all user information from text and update profile."""
    # Extract name
    name = extract_name(text)
    if name:
        profile_manager.update_profile(session_id, "name", name)
    
    # Additional info extraction could be added here (age, weight, height, goals)
    # For brevity, focusing on the name extraction which was the primary issue

# --------------------------
# Helper Functions Section
# --------------------------
def is_greeting(user_input: str) -> bool:
    """Check if the input is a greeting."""
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon"]
    normalized = user_input.strip().lower()
    
    # Check direct greeting
    if normalized in greetings:
        return True
    
    # Check if starts with a greeting
    words = normalized.split()
    if len(words) <= 3 and any(greet == words[0] for greet in greetings):
        return True
        
    return False

def is_name_query(user_input: str) -> bool:
    """Check if user is asking about their name."""
    patterns = [
        r"what(?:'s| is) my name",
        r"do you (know|remember) (me|my name)",
        r"who am i"
    ]
    
    normalized = user_input.strip().lower()
    for pattern in patterns:
        if re.search(pattern, normalized):
            return True
    return False

def is_domain_query(user_input: str) -> bool:
    """Check if the query is relevant to fitness/nutrition domains."""
    # Comprehensive list of fitness and nutrition keywords
    fitness_keywords = [
        "workout", "exercise", "training", "fitness", "cardio", "strength",
        "muscle", "weight", "lift", "squat", "deadlift", "bench", "press",
        "routine", "program", "set", "rep", "form", "technique", "progress",
        "gym", "run", "jog", "sprint", "stretch", "warmup", "cooldown", 
        "train", "fit", "body", "tone", "burn", "fat", "gain", "mass"
    ]
    
    nutrition_keywords = [
        "diet", "nutrition", "food", "meal", "eat", "protein", "carb", "fat",
        "calorie", "macro", "vitamin", "mineral", "supplement", "hydration",
        "breakfast", "lunch", "dinner", "snack", "recipe", "portion"
    ]
    
    health_keywords = [
        "health", "body", "lose", "gain", "build", "cut", "bulk",
        "sleep", "recovery", "injury", "stretch", "flexible", "mobility",
        "joint", "muscle", "sore", "pain", "posture", "form"
    ]
    
    all_keywords = fitness_keywords + nutrition_keywords + health_keywords
    
    # Check if any keyword appears in the input
    normalized = user_input.strip().lower()
    for keyword in all_keywords:
        if keyword in normalized:
            return True
    
    # Common fitness questions that might not contain keywords
    fitness_patterns = [
        r"how (should|do|can) i (train|eat)",
        r"what (should|can) i (do|eat)",
        r"how (many|much) .* (eat|consume|drink|take|lift)",
        r"best way to",
        r"tips for",
        r"advice (on|for)",
        r"recommend .* (workout|exercise|food)",
        r"help (me with|with my)",
        r"plan for",
        r"how (often|long)"
    ]
    
    for pattern in fitness_patterns:
        if re.search(pattern, normalized):
            return True
    
    return False

# --------------------------
# Response Functions
# --------------------------
def generate_greeting(session_id):
    """Generate personalized greeting."""
    profile = profile_manager.get_profile(session_id)
    name = profile.get("name")
    
    if name:
        return f"Hello {name}! I'm FeelFit, your personal fitness assistant. How can I help with your workouts, nutrition, or progress tracking today?"
    else:
        return "Hello! I'm FeelFit, your personal fitness assistant. How can I help with your workouts, nutrition, or progress tracking today?"

def generate_name_response(session_id):
    """Generate response when user asks about their name."""
    profile = profile_manager.get_profile(session_id)
    name = profile.get("name")
    
    if name:
        return f"Your name is {name}. How can I help with your fitness journey today?"
    else:
        return "I don't have your name yet. Would you like to introduce yourself? That way I can personalize our fitness conversations."

# --------------------------
# Chat Processing Function
# --------------------------
def chat_with_feelfit(user_input: str, chat_history: ChatMessageHistory, session_id: str = "default") -> str:
    """Process user input with context and chat history ensuring domain-specificity."""
    try:
        # Extract user information from input
        extract_user_info(user_input, session_id)
        
        # Handle greetings
        if is_greeting(user_input):
            greeting_response = generate_greeting(session_id)
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(greeting_response)
            return greeting_response
        
        # Handle name queries
        if is_name_query(user_input):
            name_response = generate_name_response(session_id)
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(name_response)
            return name_response

        # Check if the query is within the fitness/nutrition domain
        if not is_domain_query(user_input):
            error_msg = ("I'm sorry, but I can only assist with fitness and nutrition-related queries. "
                         "Please ask a question related to workouts, nutrition, or progress tracking.")
            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(error_msg)
            return error_msg

        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No specific context available"

        # Get user profile
        profile = profile_manager.get_profile(session_id)
        
        # Prepare input data for the chain
        input_data = {
            "input": user_input,
            "context": context,
            "history": chat_history.messages,
            "input_documents": retrieved_docs,
            "name": profile.get("name") or "Unknown",
            "age": profile.get("age") or "Unknown",
            "weight": profile.get("weight") or "Unknown",
            "height": profile.get("height") or "Unknown",
            "goal": profile.get("goal") or "Unknown"
        }

        # Generate response
        response = stuff_chain.invoke(input_data)
        response_text = response.get("text", response.get("output_text", "Could not generate response"))

        # Update chat history
        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(response_text)

        return response_text

    except Exception as e:
        print(f"Chat Error: {str(e)}")
        error_response = "I'm having trouble processing that request. Could you rephrase or provide more details?"
        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(error_response)
        return error_response