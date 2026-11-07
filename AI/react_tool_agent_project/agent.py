from tools.tools import describe_dataframe
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
class DataSummaryAgent:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.agent = create_agent(
            model=self.llm,
            tools=[describe_dataframe 
                   #load_csv, summarize_dataframe, summarize_expenses
                   ]
        )

    def run(self, input_text: str) -> dict:
        """Run the agent with the given input and return a dictionary response.
        
        Args:
            input_text (str): The user's input text/query
            
        Returns:
            dict: A dictionary containing the agent's response and status
        """
        try:
            result = self.agent.invoke(input_text)
            return {
                "output": result,
                "type": "response",
                "success": True
            }
        except Exception as e:
            return {
                "output": f"Error: {str(e)}",
                "type": "error",
                "success": False
            }