from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage

def chat_with_memory(user_input):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(return_messages=True)
    
    # 🔴 Ustaw breakpoint TUTAJ (kliknij obok numeru linii)
    history = memory.load_memory_variables({})["chat_history"]
    
    # 🔴 I tutaj
    messages = history + [HumanMessage(content=user_input)]
    
    # 🔴 I tutaj
    response = llm.invoke(messages)
    
    return response.content

# Wywołanie
result = chat_with_memory("Cześć!")
print(result)

