from langchain_openai import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate,
                               HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate,
                               PromptTemplate
)
from langchain.schema import HumanMessage, SystemMessage
import dotenv
import os 
OPEN_AI = os.getenv("OPEN_AI")

llm = ChatOpenAI(api_key = OPEN_AI,
                 model = "gpt-4o")
llm.invoke("Którą wersją czatu jestes?")
template = "Odpowiadaj w języku {jezyk}. Odpowiadaj jak {formalny}. Odpowiedz na pytanie: {pytanie}"

prompt = PromptTemplate(input_variables=['jezyk',
                                         'formalny',
                                         'pytanie'],
                        template=template)

chain = prompt  | llm 
ex_1 = {'jezyk': 'niemiecki',
        'formalny': 'pirat',
        'pytanie': 'W którym roku wydano pierwszą książkę o AI?'}
chain.invoke(ex_1)
#chain.invoke({'jezyk': 'niemieckim',
#              'formalny': 'formalny',
#              'pytanie': 'ile masz lat?'})


## z tutorials - to check

from langchain.chat_models import init_chat_models


## stworzenie klasy 

class GTPWithTemplates:
    def __init__(self, model_name):
        try:
            OPEN_AI = os.getenv("OPEN_AI")
        except:
            raise ValueError("Brak klucza OPEN_AI w zmiennych srodowiskowych")
        self.llm = init_chat_models(model = model_name)
        print(f"Połączono z {model_name}")
    def simple_template(self, message, marka):
        template  = ', odpowiedz na pytanie {message}, rekomendując tylko produkty marki: {marka}'
        prompt = PromptTemplate(input_variables=['marka','message'],
                                template = template)
        chain = prompt + self.llm
        return chain.invoke({'message': message,
                             'marka': marka})['content']


        
