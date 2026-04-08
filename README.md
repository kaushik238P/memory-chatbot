#  Memory Chatbot with LangChain

A conversational AI chatbot that remembers everything said in the session using LangChain memory.

##  How it works
User message → Added to history → LLM sees full history → Responds in context

##  Tech Stack
- LangChain (MessagesPlaceholder)
- OllamaLLM (llama3.2)
- HumanMessage + AIMessage for memory

##  Run it
pip install langchain langchain-ollama
ollama pull llama3.2
python memory_chat.py

##  Features
- Remembers name, skills, and context across turns
- Uses MessagesPlaceholder to inject history into prompt
- Prints full memory structure at the end

##  Key learning
- Without memory → LLM forgets everything between messages
- chat_history list grows with each turn
- MessagesPlaceholder injects history into the prompt automatically

##  Author
Kaushik — Mechanical Engineering @ SVNIT Surat, transitioning to AI Engineering