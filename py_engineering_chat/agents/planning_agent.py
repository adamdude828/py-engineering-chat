from typing import Dict, Any, List
import json
import os
from .base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class PlanningAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.model = ChatOpenAI(temperature=0, model="gpt-4o")
        self.plan = {}

    def create_chain(self):
        parser = StrOutputParser()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a planning assistant. Create or refine a detailed plan based on the following input:"),
            ("human", "{input}")
        ])
        
        return prompt | self.model | parser

    def generate_initial_plan(self, task_title: str, description: str) -> Dict[str, Any]:
        chain = self.create_chain()
        initial_prompt = f"Task Title: {task_title}\nDescription: {description}\n\nCreate a detailed plan with granular steps. Each step should describe file changes, types involved, new methods, inputs, and outputs. Also, include a test plan for each step."
        response = chain.invoke({"input": initial_prompt})
        
        # Parse the response and structure it into a dictionary
        # This is a simplified parsing, you might need to adjust based on the actual output
        plan = {
            "task_title": task_title,
            "description": description,
            "steps": self._parse_steps(response)
        }
        return plan

    def _parse_steps(self, response: str) -> List[Dict[str, Any]]:
        # This is a placeholder implementation. You'll need to implement proper parsing
        # based on the actual structure of the response
        steps = []
        # Parse the response and create structured steps
        return steps

    def refine_plan(self, feedback: str) -> None:
        chain = self.create_chain()
        current_plan = json.dumps(self.plan, indent=2)
        refine_prompt = f"Current plan:\n{current_plan}\n\nFeedback: {feedback}\n\nRefine the plan based on this feedback."
        response = chain.invoke({"input": refine_prompt})
        
        # Update the plan based on the refined response
        # This is a placeholder, you'll need to implement proper updating logic
        self.plan = self._parse_steps(response)

    def save_plan(self) -> None:
        shadow_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".shadow")
        os.makedirs(shadow_dir, exist_ok=True)
        plan_path = os.path.join(shadow_dir, "project_plan.json")
        with open(plan_path, "w") as f:
            json.dump(self.plan, f, indent=2)

    def chat(self):
        print("Welcome to the Planning Agent. Let's create a detailed project plan.")
        task_title = input("Enter the task title: ")
        description = input("Enter the task description: ")

        self.plan = self.generate_initial_plan(task_title, description)
        
        while True:
            print("\nCurrent Plan:")
            print(json.dumps(self.plan, indent=2))
            
            feedback = input("\nProvide feedback or type 'done' if the plan is complete: ")
            if feedback.lower() == 'done':
                break
            
            self.refine_plan(feedback)

        self.save_plan()
        print(f"\nPlan saved to {os.path.join('.shadow', 'project_plan.json')}")

def chat_with_planning_agent():
    agent = PlanningAgent()
    agent.chat()

if __name__ == "__main__":
    chat_with_planning_agent()