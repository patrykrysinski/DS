## przykład uzycia
from week_2 import MemoryAgent
chat_1 = MemoryAgent(model_name='gpt-4o-mini',
                     rola_projekt="Business consultant")

chat_1.buffer_memory()
chat_1
# rozmowa
exchanges = [
            "Cześć! Mam na imię Anna.",
            "Czy mozesz wytlumaczyć czym jest regresja liniowa?",
            "Czy to dobry model do interpretacji wyników?",
            "Jak mam na imię i jakie modele omówiliśmy?"
        ]

for i in exchanges:
    chat_1.ask_question(i)

hist = chat_1.load_memory()

hist['chat_history']

## summary version

chat_2 = MemoryAgent(model_name='gpt-4o-mini',
                     rola_projekt='Machine learning engineer')
chat_2.summary_memory()
for i in exchanges:
    chat_2.ask_question(i)

hist = chat_2.load_memory()

mes = hist['chat_history'][0]

## do sprawdzenia:

#1. jak wyciągnąć stringa z historii (tekst wiadomości)
# 2. te metody są przestarzałe, trzeba przejść na langgraph 
# https://focused-io.translate.goog/lab/customizing-memory-in-langgraph-agents-for-better-conversations?_x_tr_sl=en&_x_tr_tl=pl&_x_tr_hl=pl&_x_tr_pto=rq