import os
import csv
import json
import random
from datetime import date, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory

# ── 1. ENV & EMBEDDINGS ───────────────────────────────────────────────────────
load_dotenv()  
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

os.environ["HF_TOKEN"] = HF_TOKEN
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── 2. VECTORSTORE PERSISTENCE ───────────────────────────────────────────────
persist_dir = "vectorstore_db"
if os.path.exists(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    loader = TextLoader("datatext.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectorstore.persist()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ── 3. LLM SETUP ───────────────────────────────────────────────────────────────
llm = ChatGroq(
    groq_api_key=GROQ_KEY,
    model_name="Llama3-8b-8192",
    temperature=0.3
)

# ── 4. SYSTEM PROMPT ──────────────────────────────────────────────────────────
system_prompt = (
    "You are FeelFit, a friendly and knowledgeable AI fitness coach. "
    "You specialize in guiding users on fitness, including workouts, nutrition, diets, physical activities, and yoga. "
    "Your goal is to motivate users, answer their health-related questions with care, and provide well-informed suggestions. "
    "If someone asks a question outside of fitness, respond politely with: "
    "\"Sorry, I’m here to help with fitness only. Let’s focus on health and workouts!\"\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

history = ChatMessageHistory()
llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    input_key="input_documents"
)

# ── 5. CHAT FUNCTION ──────────────────────────────────────────────────────────
def chat_fitness(user_input: str) -> str:
    docs = retriever.invoke(user_input)
    context = "\n\n".join(d.page_content for d in docs) if docs else ""
    payload = {
        "input": user_input,
        "context": context,
        "history": history.messages,
        "input_documents": docs
    }
    try:
        resp = stuff_chain.invoke(payload)
        text = resp.get("output_text") or resp.get("text") or str(resp)
    except Exception:
        text = "Sorry, I’m here to help with fitness only. Let’s focus on health and workouts!"
    history.add_user_message(user_input)
    history.add_ai_message(text)
    return text

# ── 6. PROGRESS TRACKING & DASHBOARD ────────────────────────────────────────
PROGRESS_FILE = "progress_log.csv"

def log_progress(metric: str, value: float):
    file_exists = os.path.isfile(PROGRESS_FILE)
    with open(PROGRESS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "metric", "value"])
        writer.writerow([date.today().isoformat(), metric, value])

def plot_progress():
    data = {}
    with open(PROGRESS_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["metric"]
            v = float(row["value"])
            dt = row["date"]
            data.setdefault(m, []).append((dt, v))
    plt.figure(figsize=(8,4))
    for metric, vals in data.items():
        dates, vs = zip(*vals)
        plt.plot(dates, vs, label=metric)
    plt.title("Your Progress Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("progress.png")
    plt.close()
    print("📈 Progress chart saved to progress.png")

# ── 7. DYNAMIC WORKOUT BUILDER ───────────────────────────────────────────────
EX_DB = [
    {"name": "Push-ups",      "duration": 5, "equipment": False},
    {"name": "Squats",        "duration": 5, "equipment": False},
    {"name": "Jumping Jacks", "duration": 3, "equipment": False},
    {"name": "Plank",         "duration": 4, "equipment": False},
    {"name": "Dumbbell Curl", "duration": 5, "equipment": True},
    {"name": "Burpees",       "duration": 5, "equipment": False},
]

def build_workout(minutes: int, has_equipment: bool):
    pool = [e for e in EX_DB if has_equipment or not e["equipment"]]
    random.shuffle(pool)
    plan, used = [], 0
    for ex in pool:
        if used + ex["duration"] <= minutes:
            plan.append(ex)
            used += ex["duration"]
        if used >= minutes:
            break
    print(f"\n🗒️ Your {minutes}-minute workout:")
    for i, ex in enumerate(plan,1):
        tag = "with equipment" if ex["equipment"] else "bodyweight"
        print(f"  {i}. {ex['name']} – {ex['duration']} min ({tag})")
    print()

# ── 8. STREAK & BADGES ───────────────────────────────────────────────────────
STREAK_FILE = "streak.json"
BADGES = {3:"🔥 3-Day Streak!",7:"🏆 1-Week Streak!",30:"💪 1-Month Streak!"}

def update_and_show_streak():
    today = date.today().isoformat()
    if os.path.exists(STREAK_FILE):
        data = json.load(open(STREAK_FILE))
    else:
        data = {"last_date":"", "streak":0}
    last = date.fromisoformat(data["last_date"]) if data["last_date"] else None
    if last == date.today() - timedelta(days=1):
        data["streak"] += 1
    elif last != date.today():
        data["streak"] = 1
    data["last_date"] = today
    json.dump(data, open(STREAK_FILE,"w"))
    s = data["streak"]
    print(f"🔥 Current streak: {s} day(s)")
    if s in BADGES:
        print(BADGES[s],"\n")

# ── 9. MAIN CLI LOOP ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🏋️‍♀️ FeelFit Chatbot + Extras is live! Type 'exit' to quit.\n")
    update_and_show_streak()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit","quit","bye"}:
            print("🏁 Keep pushing! Goodbye!")
            break

        if user_input.lower().startswith("log "):
            try:
                _, metric, val = user_input.split(maxsplit=2)
                log_progress(metric, float(val))
                print(f"✅ Logged {metric} = {val}")
                plot_progress()
            except Exception:
                print("⚠️ Usage: log <metric> <value>")
            continue

        if user_input.lower().startswith("workout"):
            try:
                _, mins, eq = user_input.split()
                build_workout(int(mins), eq.lower() in {"yes","y","true"})
            except Exception:
                print("⚠️ Usage: workout <minutes> <yes|no>")
            continue

        # In your main CLI loop, before falling back to chat_fitness:
        if "transformation" in user_input.lower():
            # 1️⃣ Gather basic details
            print("🏷️  Let’s get started on your transformation journey! May I know your:")
            age    = input("  – Age (years): ").strip()
            height = input("  – Height (cm): ").strip()
            weight = input("  – Weight (kg): ").strip()
            goal   = input("  – Primary goal (e.g. fat loss, muscle gain): ").strip()
            
            # 2️⃣ Store in context/history for later
            profile = {
                "age": age,
                "height_cm": height,
                "weight_kg": weight,
                "goal": goal
            }
            history.add_user_message(f"Profile: {profile}")
            
            # 3️⃣ Hand off to LLM with profile in the prompt
            prompt_text = (
                f"I'm a {age}-year-old, {height} cm tall, {weight} kg looking for {goal}. "
                "Please craft a 4-week fitness + nutrition plan."
            )
            reply = chat_fitness(prompt_text)
            print(f"\nFeelFit: {reply}\n")
            continue

        # fallback to AI fitness advice
        reply = chat_fitness(user_input)
        print(f"\nFeelFit: {reply}\n")