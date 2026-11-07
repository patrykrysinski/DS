import openai
import os
## 
client= openai.OpenAI(api_key = os.getenv("OPEN_AI"))

models = client.models.list()
models_id = [i.id for i in models]
response = client.responses.create(model= 'gpt-4.1-nano',
                                   input = "Cześć, co potrafisz")
print(response.output_text)

def create_response(input, model = 'gpt-4.1-nano'):
    client= openai.OpenAI(api_key = os.getenv("OPEN_AI"))
    response = client.responses.create(model= model,
                                   input = input)
    if response.error:
        return 'Mamy problem'
    return response.output_text

create_response("cześć, co tam?")

response.to_dict()

client.chat.completions.create()