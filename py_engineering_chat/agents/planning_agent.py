from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import json
import os
import time
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager

class TaskPlan(BaseModel):
    """Represents a generic task plan."""
    task_description: str = Field(description="Brief description of the task")
    objectives: List[str] = Field(description="List of objectives for the task")
    steps: List[Dict[str, Any]] = Field(description="Detailed steps to accomplish the task")
    resources: List[str] = Field(description="Resources needed for the task")
    potential_challenges: List[str] = Field(description="Potential challenges and mitigation strategies")
    additional_notes: str = Field(description="Any additional relevant information")

class PlanningAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview")
        self.conversation_history = []
        self.reserve_questions = [
            "What are the specific goals or outcomes for this task?",
            "Are there any deadlines or time constraints?",
            "What resources (people, tools, information) might be needed?",
            "Are there any potential obstacles or challenges to consider?",
            "How will progress or success be measured?",
            "Are there any dependencies on other tasks or people?",
            "What are the priority levels for different aspects of the task?",
            "Are there any specific methods or approaches that should be used?",
            "What are the potential risks associated with this task?",
            "Is there any background information or context that's important to know?"
        ]

    def ask_question(self, question):
        response = self.llm.invoke(self.conversation_history + [HumanMessage(content=question)])
        self.conversation_history.extend([HumanMessage(content=question), AIMessage(content=response.content)])
        return response.content

    def gather_task_info(self):
        initial_question = "Please describe the task you need help planning in detail."
        task_description = self.ask_question(initial_question)
        
        # Follow-up with clarifying questions based on the initial description
        clarifying_questions = self.generate_clarifying_questions(task_description)
        for question in clarifying_questions:
            self.ask_question(question)

        return task_description

    def generate_clarifying_questions(self, task_description):
        prompt = f"""Based on the following task description, generate 3-5 specific clarifying questions to gather more details. 
        Task description: {task_description}
        Do not ask about anything already clearly addressed in the description."""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        questions = response.content.split('\n')
        return [q.strip() for q in questions if q.strip()]

    def create_initial_plan(self):
        prompt = """Based on our conversation so far, create an initial task plan with the following components:
        1. Objectives
        2. Steps
        3. Resources needed
        4. Potential challenges
        5. Any additional notes
        
        Format the response as a JSON object with these keys: objectives, steps, resources, potential_challenges, additional_notes"""
        
        response = self.llm.invoke(self.conversation_history + [HumanMessage(content=prompt)])
        plan_dict = json.loads(response.content)
        return TaskPlan(task_description=self.conversation_history[1].content, **plan_dict)

    def refine_plan(self, plan):
        prompt = f"""Review the following task plan and suggest improvements or additions based on these questions:
        {self.reserve_questions}
        
        Current plan:
        {plan.json(indent=2)}
        
        Provide your suggestions as a JSON object with the same structure as the current plan."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        refined_plan_dict = json.loads(response.content)
        return TaskPlan(**refined_plan_dict)

    def generate_plan(self):
        self.gather_task_info()
        initial_plan = self.create_initial_plan()
        refined_plan = self.refine_plan(initial_plan)
        return refined_plan

    def save_plan(self, plan, filename):
        shadow_dir = ChatSettingsManager().get_project_shadow_directory()
        file_path = os.path.join(shadow_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(plan.dict(), f, indent=2)
        return f"Plan saved to {file_path}"

async def plan_step(state):
    planning_agent = PlanningAgent()
    try:
        plan = planning_agent.generate_plan()
        filename = f"task_plan_{int(time.time())}.json"
        save_message = planning_agent.save_plan(plan, filename)
        return {"plan": plan.dict(), "save_message": save_message, "error": ""}
    except Exception as e:
        return {"plan": {}, "save_message": "", "error": str(e)}
