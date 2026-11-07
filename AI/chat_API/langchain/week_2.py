from langchain_openai import ChatOpenAI
from langchain.memory import (
    ConversationBufferMemory, # zapamiętuje wszystko
    ConversationBufferWindowMemory, # zapamiętuje n poprzednich wiadomości
    ConversationSummaryMemory, # tworzy podsumowania
    ConversationSummaryBufferMemory,
    ConversationTokenBufferMemory # limit tokenów
)
from langchain.chains import ConversationChain ## to jest przestarzałe

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os 

OPEN_AI_KEY = os.getenv("OPEN_AI")

class MemoryAgent:
    def __init__(self, model_name, rola_projekt):
        self.model_name = model_name
        self.llm = ChatOpenAI(api_key=OPEN_AI_KEY,
                              model=model_name)
        self.rola_projekt = rola_projekt
        self.template = ChatPromptTemplate.from_messages([
            ("system","Jesteś eskpertem Data Science. Odpowiedz tak, aby zrozumiała to osoba na stanowisku {rola_projekt}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")

        ])
        self.chain = (RunnablePassthrough.assign(chat_history = lambda x: self.memory.load_memory_variables({})["chat_history"]
        )
        | self.template
        | self.llm
        | StrOutputParser()
        )

    def buffer_memory(self):

        self.memory = ConversationBufferMemory(memory_key="chat_history",
                                           return_messages=True)
        self.memory_type = 'Buffer Memory'
    def summary_memory(self):
        self.memory = ConversationSummaryMemory(memory_key="chat_history",
                                                return_messages=True,
                                                llm= self.llm,
                                                max_token_limit = 100)
        self.memory_type = 'Summary Memory'

    def create_chain(self):
        pass

        
    def ask_question(self,prompt):
        response = self.chain.invoke({"input": prompt,
                                      "rola_projekt": self.rola_projekt})
        self.memory.save_context({"input": prompt},
                                 {"output": response})
        print(f"Pytanie: {prompt}")
        print(f"Odpowiedź: {response}")
        return response
    def load_memory(self):
        history = self.memory.load_memory_variables({})
        print(f"Liczba wiadomości: {len(history['chat_history'])}")
        return history
    

## Week 2 z langgraph

from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # Pamięć w RAM
from langgraph.checkpoint.sqlite import SqliteSaver  # Pamięć w SQLite
from langgraph.graph.message import add_messages

class ChatState():
    messages: Annotated[Sequence[BaseMessage], add_messages]
