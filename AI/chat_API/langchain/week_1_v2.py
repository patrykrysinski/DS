import os

OPEN_AI_KEY = os.getenv("OPEN_AI")

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=1,
                 model = 'gpt-4o',
                 api_key=OPEN_AI_KEY)

llm.invoke("Jaką wersją modelu jesteś?")
llm.invoke("Która ksiązka mickiewicza jest najlepszA?")

answer = llm.invoke("ile jest 2+2?")

answer.content

class AgentWeek1:
    def __init__(self, model_name):
        self.api_key = os.getenv("OPEN_AI")
        self.model_name = model_name
        try:
            model = ChatOpenAI(api_key=self.api_key,
                           model = self.model_name)
        except:
            raise ValueError(f"Can't connect to chat using model {self.model_name} and saved API KEY.")
    def simpletemplate(model_do_opisu, rola):
        return "Jako Data Scientist opisz model {model_do_opisu}, tak aby zrozumiała to osoba na stanowisku {rola}."
    
