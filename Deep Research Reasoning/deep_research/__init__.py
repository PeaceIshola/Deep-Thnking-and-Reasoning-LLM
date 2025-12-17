"""
Deep Research Reasoning System

A modular library for advanced reasoning techniques including Chain-of-Thought,
Self-Consistency, Sequential Revision, Tree-of-Thoughts, and Deep Research Agents.
"""

__version__ = "1.0.0"

# Import main classes and functions for easy access
from .config import Config, get_client
from .cot_methods import FewShotCoT, ZeroShotCoT, GPT2CoT, few_shot_cot, zero_shot_cot
from .self_consistency import SelfConsistency, self_consistent_answer
from .sequential_revision import SequentialRevision, sequential_revision
from .tree_of_thoughts import (
    WordLadder,
    GenericToT,
    solve_word_ladder,
    tree_of_thoughts_search
)
from .deep_research_agent import (
    WebSearchTool,
    DeepResearchAgent,
    SimpleSearchAgent,
    create_research_agent,
    quick_research
)

__all__ = [
    # Config
    "Config",
    "get_client",
    
    # Chain-of-Thought
    "FewShotCoT",
    "ZeroShotCoT",
    "GPT2CoT",
    "few_shot_cot",
    "zero_shot_cot",
    
    # Self-Consistency
    "SelfConsistency",
    "self_consistent_answer",
    
    # Sequential Revision
    "SequentialRevision",
    "sequential_revision",
    
    # Tree-of-Thoughts
    "WordLadder",
    "GenericToT",
    "solve_word_ladder",
    "tree_of_thoughts_search",
    
    # Deep Research Agent
    "WebSearchTool",
    "DeepResearchAgent",
    "SimpleSearchAgent",
    "create_research_agent",
    "quick_research",
]
