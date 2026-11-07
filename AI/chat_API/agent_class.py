import os
import openai
import joblib
import datetime
api_key = os.getenv("OPEN_AI") 
class AgentDoProduktu:
    def __init__(self,name, goal, role, model='gpt-4.1-nano'):
        self.name = name
        self.goal = goal
        self.client = openai.OpenAI(api_key=api_key)
        self.role = role 
        self.model = model 
        self.memory = []
    def chat(self, message, save_answer= True):
        prompt_1 = f"Jesteś agentem AI. Cel agenta {self.goal}"
        message_hist = [{"role": "system",
                      "content": prompt_1}]
        for i in self.memory:
            message_hist.append({"role": "user",
                                 "content": i['user']})
            message_hist.append({"role": "assistant",
                                 "content": i['assistant']})
        message_hist.append({"role": self.role,
                       "content": message           
                      })
        response = self.client.chat.completions.create(
            model = self.model,
            messages=message_hist
        )
        answer = response.choices[0].message.content
        if save_answer:
            self.memory.append({self.role:message,
                            "assistant": answer})
        return answer
    def save_chat(self,path=None):
        if not path:
            time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join(os.getcwd(),f"chat_{time}.joblib")
        try:
            joblib.dump(self.memory,path)
        except Exception as e:
            print(f"Błąd podczas zapisu rozmowy: {e}")
    def load_chat(self,path):
        try:
            self.memory = joblib.load(path)
        except:
            raise("Nie udało się załadować rozmowy.")
    def make_summary(self):
        return self.chat(f"Podsumuj poniższą rozmowę, krótkim streszczeniem: {self.memory}",save_answer=False)
        