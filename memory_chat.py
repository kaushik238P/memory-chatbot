from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ---- Setup ----
llm = OllamaLLM(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI placement assistant helping 
     Kaushik prepare for AI job interviews. 
     Keep answers concise and practical."""),
    MessagesPlaceholder(variable_name="history"),  # ← history goes here
    ("human", "{input}")
])

chain = prompt | llm

# ---- Memory storage (simple list) ----
chat_history = []

def chat(user_message):
    print(f"\nYou: {user_message}")
    
    # Run chain with history
    response = chain.invoke({
        "history": chat_history,
        "input": user_message
    })
    
    # Save to history
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response))
    
    print(f"AI: {response}")
    return response

# ---- Simulate a real conversation ----
print("=" * 50)
print("CONVERSATION WITH MEMORY")
print("=" * 50)

# Turn 1 — introduce yourself
chat("Hi! My name is Kaushik and I study at SVNIT.")

# Turn 2 — test if it remembers
chat("What is my name and where do I study?")

# Turn 3 — add more context
chat("I want to get placed in an AI company. My skills are Python and LangChain.")

# Turn 4 — test memory of skills
chat("Based on what I told you, what AI role suits me best?")

# Turn 5 — ask for interview prep
chat("Give me one interview question based on my skills.")

# Print memory stats
print("\n" + "=" * 50)
print(f"Total messages in memory: {len(chat_history)}")
print(f"Memory structure:")
for i, msg in enumerate(chat_history):
    role = "You" if isinstance(msg, HumanMessage) else "AI"
    preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
    print(f"  {i+1}. {role}: {preview}")