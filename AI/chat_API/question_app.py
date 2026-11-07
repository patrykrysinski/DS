import openai
import os
import joblib
client = openai.OpenAI(api_key=os.getenv('OPEN_AI'))
def main():
    modele_opisy = {}
    questions_logs = {}
    print("Witaj w moim API do zadawania pytań w czacie GPT.")
    change_model = input("Używamy chatu gpt4.1-nano. Czy chcesz zmienić model? (Y/N)")
    id = 0
    while True:

        if change_model.lower()== 'y':
            models = client.models.list()
            models_id = [i.id for i in models]
            no_models = len(models_id)
            input(f"Obecnie dostąpne są {no_models} modele. Szczegowe opcje nazewnictwa możesz znaleźć tutaj: https://platform.openai.com/docs/pricing /n Kliknij enter, aby przejść dalej. ")
  
            for i in range(0,no_models,10):
                q1 = input("Czy chcesz poznac listę 10 nazw modeli? (Y/N)")
                if q1.lower()=='y':
                    if i+10 > no_models:
                        print(f'{models_id[i:]}')
                    else:
                        print(f'{models_id[i:i+10]}')
                else:
                    break
            
            modele = input("Podaj nazwe modelu, którego chcesz poznać opis. Wciśnij 'N', aby zakończyć.")
            nazwy_modeli_gpt = client.responses.create(model = 'gpt-4.1-nano',
                                                       input = f'Opisz model {modele}'
                                                       )
            input(nazwy_modeli_gpt.output_text)
            nazwy_modeli_gpt = nazwy_modeli_gpt.to_dict()
            nazwy_modeli_gpt['model'] = modele
            modele_opisy[id] = nazwy_modeli_gpt
            model = input("Podaj nazwę modelu, którego chcesz uzywać. ")
            break
        elif change_model.lower() =='n':
            model = 'gpt-4.1-nano'
        else:
            input("Podano niepoprawną wartość.")
        question = input('Zadaj pytanie modelowi')
        answer = client.responses.create(model=  model,
                                         input = question)
        input(answer.output_text)
        answer = answer.to_dict()
        answer['question'] = question 
        questions_logs[id] = answer
        id +=1
        q3 = input("Czy chcesz zadać kolejne pytanie (y/n)?")
        if q3.lower() !='y':
            break
    joblib.dump(questions_logs,"q_logs.joblib")
    try:
        joblib.dump(nazwy_modeli_gpt,"nazwy_modeli.joblib")
    except:
        pass
main()