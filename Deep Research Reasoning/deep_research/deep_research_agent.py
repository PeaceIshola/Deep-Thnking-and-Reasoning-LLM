"""
Deep Research Agent module combining reasoning with web search.
Uses ReAct pattern for multi-step research and reasoning.
"""

from typing import Optional
from duckduckgo_search import DDGS
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama


class WebSearchTool:
    """Web search tool using DuckDuckGo."""
    
    @staticmethod
    def search(query: str, k: int = 5) -> str:
        """
        Search the web and return concatenated snippets.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            str: Concatenated search results
        """
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=k))
            
            if not results:
                return f"No search results found for: {query}"
            
            snippets = []
            for r in results:
                title = r.get('title', 'No title')
                body = r.get('body', 'No description')
                snippets.append(f"{title}: {body}")
            
            return "\n\n".join(snippets)
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def as_langchain_tool(self) -> Tool:
        """
        Convert to LangChain Tool.
        
        Returns:
            Tool: LangChain tool wrapper
        """
        return Tool(
            name="DuckDuckGo Search",
            func=self.search,
            description="Search the public web. Input: a plain English query. Returns: concatenated snippets."
        )


class DeepResearchAgent:
    """Deep research agent with reasoning and web search capabilities."""
    
    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        temperature: float = 0.7,
        search_results: int = 5
    ):
        """
        Initialize Deep Research Agent.
        
        Args:
            model: Reasoning model to use
            temperature: Sampling temperature
            search_results: Number of search results per query
        """
        self.model = model
        self.temperature = temperature
        self.search_results = search_results
        
        # Initialize LLM
        self.llm = ChatOllama(model=model, temperature=temperature)
        
        # Initialize search tool
        self.search_tool = WebSearchTool()
        
        # Initialize agent
        self.agent = initialize_agent(
            [self.search_tool.as_langchain_tool()],
            self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )
    
    def research(self, question: str) -> str:
        """
        Research a question using web search and reasoning.
        
        Args:
            question: Question to research
            
        Returns:
            str: Research answer
        """
        return self.agent.run(question)
    
    def research_with_context(
        self,
        question: str,
        context: Optional[str] = None
    ) -> str:
        """
        Research with additional context.
        
        Args:
            question: Question to research
            context: Optional additional context
            
        Returns:
            str: Research answer
        """
        if context:
            full_question = f"Context: {context}\n\nQuestion: {question}"
        else:
            full_question = question
        
        return self.research(full_question)


class SimpleSearchAgent:
    """Simpler agent without LangChain for direct control."""
    
    def __init__(
        self,
        client,
        model: str = "deepseek-r1:8b",
        max_iterations: int = 5,
        search_results: int = 5
    ):
        """
        Initialize Simple Search Agent.
        
        Args:
            client: OpenAI client instance
            model: Model name
            max_iterations: Maximum research iterations
            search_results: Number of search results per query
        """
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.search_results = search_results
        self.search_tool = WebSearchTool()
    
    def research(self, question: str, verbose: bool = True) -> str:
        """
        Research a question with manual ReAct loop.
        
        Args:
            question: Question to research
            verbose: If True, print thinking process
            
        Returns:
            str: Final answer
        """
        conversation = [
            {"role": "system", "content": """You are a research assistant. To answer questions:
1. Think about what information you need
2. Use [SEARCH: query] to search the web
3. Analyze the results
4. Either answer or search again
5. When ready, provide [ANSWER: your final answer]"""},
            {"role": "user", "content": question}
        ]
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")
            
            # Get model response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                temperature=0.7,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            if verbose:
                print(f"Agent: {content[:200]}...")
            
            # Check for search command
            if "[SEARCH:" in content:
                search_query = content.split("[SEARCH:")[1].split("]")[0].strip()
                if verbose:
                    print(f"Searching: {search_query}")
                
                search_results = self.search_tool.search(search_query, self.search_results)
                
                conversation.append({"role": "assistant", "content": content})
                conversation.append({"role": "user", "content": f"Search results:\n{search_results}"})
                
                continue
            
            # Check for final answer
            if "[ANSWER:" in content:
                answer = content.split("[ANSWER:")[1].split("]")[0].strip()
                if verbose:
                    print(f"\nFinal Answer: {answer}")
                return answer
            
            # If no special command, treat as final answer
            return content
        
        return "Unable to find answer within iteration limit."


# Convenience functions
def create_research_agent(
    model: str = "deepseek-r1:8b",
    temperature: float = 0.7
) -> DeepResearchAgent:
    """
    Create a deep research agent with default settings.
    
    Args:
        model: Model name
        temperature: Sampling temperature
        
    Returns:
        DeepResearchAgent: Configured agent
    """
    return DeepResearchAgent(model=model, temperature=temperature)


def quick_research(
    question: str,
    model: str = "deepseek-r1:8b"
) -> str:
    """
    Quick research function for simple queries.
    
    Args:
        question: Question to research
        model: Model name
        
    Returns:
        str: Answer
    """
    agent = DeepResearchAgent(model=model)
    return agent.research(question)
