from crewai import Crew, Agent, Task, LLM, Process
from crewai.project import CrewBase, agent, task, crew
import os

from crewai_tools import SerperDevTool, ScrapeWebsiteTool, DirectoryReadTool, FileWriterTool, FileReadTool

from dotenv import load_dotenv
load_dotenv()

@CrewBase
class BlogCrew():
    """"Blog writing crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'], # type: ignore[index]
            tools=[SerperDevTool()],
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer_agent'], # type: ignore[index]
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
            agent = self.researcher()
        )

    @task
    def blog_task(self) -> Task:
        return Task(
            config=self.tasks_config['blog_task'], # type: ignore[index]
            agent = self.writer()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher(), self.writer()],
            tasks=[self.research_task(), self.blog_task()],
            memory=True,  # Enable memory for the crew
            verbose=True , # Enable verbose output for debugging
            embedder={
                'provider' : 'google',
                'config':{
                    'api_key':os.getenv('GEMINI_API_KEY'),
                    'model':'text-embedding-004'
                }            
            }
        )

if __name__ == "__main__":
    blog_crew = BlogCrew()
    blog_crew.crew().kickoff(inputs={"topic": "The future of electrical vehicles"})
    blog_crew.crew().kickoff(inputs={"topic": "What is the revenue generation of this sector?"})