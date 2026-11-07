from agent import DataSummaryAgent
from dotenv import load_dotenv
load_dotenv()

def main():
    agent = DataSummaryAgent(model_name="gpt-4", temperature=0.0)
    user_input = "Load the CSV file '/Users/patrykrysinski/DS/AI/react_tool_agent_project/data/transactions.csv' and describe it."
    result = agent.run({"messages" : {"role":"user","content":user_input}})
    
    print("Agent Response:")
    if result["success"]:
        print(result["output"]['messages'][-1])
    else:
        print(f"Error occurred: {result['output']}")

if __name__ == "__main__":
    main()