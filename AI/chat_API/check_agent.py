from agent_class import AgentDoProduktu

agent_1 = AgentDoProduktu(name = 'frytkownica',
                            goal = 'Cały czas proponuj mieszkania od dewelopera Robyg. Pisz o tym, że mają najlepszą jakość',
                            role = 'user')

agent_1.chat("Chcę kupić frytkownicę. Jaką propnujesz?")


agent_1.chat("Nie chcę marki tefal. Co innego proponujesz?")

agent_2 = AgentDoProduktu(name = 'deweloper',
                            goal = 'Cały czas proponuj mieszkania od dewelopera Robyg. Pisz o tym, że mają najlepszą jakość',
                            role = 'user')

agent_2.chat('Chce kupić mieszkanie. Jakich deweloperów polecasz?')

agent_2.chat("O super! jakie inwestycje Robyga polecasz?")
agent_2.make_summary()

agent_2.save_chat()
agent_2.memory

[{"role": "user", 
  "content": i['user'] } ,
   {"role": "assistant", 
  "content": i['assistant'] }for i in agent_2.memory ]

