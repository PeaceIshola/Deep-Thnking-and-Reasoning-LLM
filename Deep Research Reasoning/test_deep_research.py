"""
Comprehensive test file for the Deep Research Reasoning System.
Tests all major components and their functionality.
"""

import sys
import os
from openai import OpenAI

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from deep_research.config import get_client, Config
from deep_research.cot_methods import FewShotCoT, ZeroShotCoT, GPT2CoT
from deep_research.self_consistency import self_consistent_answer
from deep_research.sequential_revision import sequential_revision
from deep_research.tree_of_thoughts import solve_word_ladder, tree_of_thoughts_search
from deep_research.deep_research_agent import WebSearchTool, SimpleSearchAgent


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []
    
    def add_pass(self, test_name):
        self.passed.append(test_name)
        print(f"✓ PASS: {test_name}")
    
    def add_fail(self, test_name, error):
        self.failed.append((test_name, error))
        print(f"✗ FAIL: {test_name} - {error}")
    
    def add_skip(self, test_name, reason):
        self.skipped.append((test_name, reason))
        print(f"⊘ SKIP: {test_name} - {reason}")
    
    def summary(self):
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Passed: {len(self.passed)}")
        print(f"Failed: {len(self.failed)}")
        print(f"Skipped: {len(self.skipped)}")
        print(f"Total: {len(self.passed) + len(self.failed) + len(self.skipped)}")
        
        if self.failed:
            print("\nFailed Tests:")
            for test_name, error in self.failed:
                print(f"  - {test_name}: {error}")
        
        return len(self.failed) == 0


def test_config():
    """Test configuration and client setup."""
    print("\n" + "="*70)
    print("TESTING: Configuration Module")
    print("="*70)
    results = TestResults()
    
    # Test Ollama client creation
    try:
        client = Config.get_ollama_client()
        assert isinstance(client, OpenAI)
        results.add_pass("Ollama client creation")
    except Exception as e:
        results.add_fail("Ollama client creation", str(e))
    
    # Test get_client function
    try:
        client = get_client(use_openai=False)
        assert isinstance(client, OpenAI)
        results.add_pass("get_client function")
    except Exception as e:
        results.add_fail("get_client function", str(e))
    
    return results


def test_word_ladder():
    """Test word ladder solver."""
    print("\n" + "="*70)
    print("TESTING: Word Ladder (Tree-of-Thoughts)")
    print("="*70)
    results = TestResults()
    
    try:
        vocab = {"hit", "dot", "cog", "log", "dog", "lot", "lit", "hot"}
        start = "hit"
        goal = "cog"
        
        result = solve_word_ladder(start, goal, vocab)
        
        if result and result[0] == start and result[-1] == goal:
            results.add_pass("Word ladder solver")
            print(f"  Solution: {' -> '.join(result)}")
        else:
            results.add_fail("Word ladder solver", "Invalid solution path")
    except Exception as e:
        results.add_fail("Word ladder solver", str(e))
    
    return results


def test_gpt2_cot():
    """Test GPT-2 Chain-of-Thought."""
    print("\n" + "="*70)
    print("TESTING: GPT-2 Chain-of-Thought")
    print("="*70)
    results = TestResults()
    
    try:
        cot = GPT2CoT()
        question = "If 2 + 2 = 4, what is 3 + 3?"
        
        # Test greedy decoding
        answer = cot.generate(question, decoding_strategy="greedy", max_new_tokens=30)
        if answer and len(answer) > 0:
            results.add_pass("GPT-2 CoT greedy decoding")
        else:
            results.add_fail("GPT-2 CoT greedy decoding", "No output generated")
        
        # Test top-k decoding
        answer = cot.generate(question, decoding_strategy="top_k", max_new_tokens=30)
        if answer and len(answer) > 0:
            results.add_pass("GPT-2 CoT top-k decoding")
        else:
            results.add_fail("GPT-2 CoT top-k decoding", "No output generated")
        
    except Exception as e:
        results.add_fail("GPT-2 CoT", str(e))
    
    return results


def test_few_shot_cot(client):
    """Test Few-Shot Chain-of-Thought."""
    print("\n" + "="*70)
    print("TESTING: Few-Shot Chain-of-Thought")
    print("="*70)
    results = TestResults()
    
    try:
        # Test with Ollama availability
        cot = FewShotCoT(client, Config.DEFAULT_BASE_MODEL)
        question = "If 5 apples cost $10, how much do 3 apples cost?"
        
        answer = cot.generate(question)
        if answer and len(answer) > 0:
            results.add_pass("Few-Shot CoT generation")
            print(f"  Answer preview: {answer[:100]}...")
        else:
            results.add_fail("Few-Shot CoT generation", "No output generated")
    except Exception as e:
        # If Ollama is not running, skip this test
        if "Connection" in str(e) or "Failed to connect" in str(e):
            results.add_skip("Few-Shot CoT", "Ollama not running")
        else:
            results.add_fail("Few-Shot CoT", str(e))
    
    return results


def test_zero_shot_cot(client):
    """Test Zero-Shot Chain-of-Thought."""
    print("\n" + "="*70)
    print("TESTING: Zero-Shot Chain-of-Thought")
    print("="*70)
    results = TestResults()
    
    try:
        cot = ZeroShotCoT(client, Config.DEFAULT_BASE_MODEL)
        question = "What is 12 divided by 4?"
        
        answer = cot.generate(question)
        if answer and len(answer) > 0:
            results.add_pass("Zero-Shot CoT generation")
            print(f"  Answer preview: {answer[:100]}...")
        else:
            results.add_fail("Zero-Shot CoT generation", "No output generated")
    except Exception as e:
        if "Connection" in str(e) or "Failed to connect" in str(e):
            results.add_skip("Zero-Shot CoT", "Ollama not running")
        else:
            results.add_fail("Zero-Shot CoT", str(e))
    
    return results


def test_self_consistency(client):
    """Test Self-Consistency."""
    print("\n" + "="*70)
    print("TESTING: Self-Consistency")
    print("="*70)
    results = TestResults()
    
    try:
        question = "What is 2 + 2?"
        winner, votes = self_consistent_answer(client, question, n=3, model=Config.DEFAULT_BASE_MODEL)
        
        if winner and votes:
            results.add_pass("Self-Consistency generation")
            print(f"  Winning answer: {winner}")
            print(f"  Vote distribution: {votes}")
        else:
            results.add_fail("Self-Consistency generation", "No output generated")
    except Exception as e:
        if "Connection" in str(e) or "Failed to connect" in str(e):
            results.add_skip("Self-Consistency", "Ollama not running")
        else:
            results.add_fail("Self-Consistency", str(e))
    
    return results


def test_sequential_revision(client):
    """Test Sequential Revision."""
    print("\n" + "="*70)
    print("TESTING: Sequential Revision")
    print("="*70)
    results = TestResults()
    
    try:
        question = "What is the capital of France?"
        final_answer = sequential_revision(client, question, max_steps=2, model=Config.DEFAULT_BASE_MODEL, verbose=False)
        
        if final_answer and len(final_answer) > 0:
            results.add_pass("Sequential Revision generation")
            print(f"  Final answer preview: {final_answer[:100]}...")
        else:
            results.add_fail("Sequential Revision generation", "No output generated")
    except Exception as e:
        if "Connection" in str(e) or "Failed to connect" in str(e):
            results.add_skip("Sequential Revision", "Ollama not running")
        else:
            results.add_fail("Sequential Revision", str(e))
    
    return results


def test_tree_of_thoughts(client):
    """Test Tree-of-Thoughts."""
    print("\n" + "="*70)
    print("TESTING: Tree-of-Thoughts Search")
    print("="*70)
    results = TestResults()
    
    try:
        question = "List three colors in the rainbow."
        solution, score = tree_of_thoughts_search(
            client, 
            question, 
            depth=1, 
            width=2, 
            model=Config.DEFAULT_BASE_MODEL, 
            verbose=False
        )
        
        if solution:
            results.add_pass("Tree-of-Thoughts search")
            print(f"  Solution (score {score}): {solution[:100]}...")
        else:
            results.add_fail("Tree-of-Thoughts search", "No solution generated")
    except Exception as e:
        if "Connection" in str(e) or "Failed to connect" in str(e):
            results.add_skip("Tree-of-Thoughts", "Ollama not running")
        else:
            results.add_fail("Tree-of-Thoughts", str(e))
    
    return results


def test_web_search_tool():
    """Test Web Search Tool."""
    print("\n" + "="*70)
    print("TESTING: Web Search Tool")
    print("="*70)
    results = TestResults()
    
    try:
        tool = WebSearchTool()
        result = tool.search("Python programming", k=2)
        
        # Result should at minimum return a string (even if no results found)
        if result and isinstance(result, str):
            results.add_pass("Web search tool")
            print(f"  Search result preview: {result[:100]}...")
        else:
            results.add_fail("Web search tool", "No search results")
    except Exception as e:
        results.add_fail("Web search tool", str(e))
    
    return results


def test_simple_search_agent(client):
    """Test Simple Search Agent."""
    print("\n" + "="*70)
    print("TESTING: Simple Search Agent")
    print("="*70)
    results = TestResults()
    
    try:
        agent = SimpleSearchAgent(client, Config.DEFAULT_REASONING_MODEL, max_iterations=1)
        question = "What is the weather like?"
        answer = agent.research(question, verbose=False)
        
        if answer and len(answer) > 0:
            results.add_pass("Simple search agent")
            print(f"  Answer preview: {answer[:100]}...")
        else:
            results.add_fail("Simple search agent", "No output generated")
    except Exception as e:
        if "Connection" in str(e) or "Failed to connect" in str(e):
            results.add_skip("Simple search agent", "Ollama not running")
        else:
            results.add_fail("Simple search agent", str(e))
    
    return results


def main():
    """Run all tests."""
    print("\n" + "#"*70)
    print("# DEEP RESEARCH REASONING SYSTEM - TEST SUITE")
    print("#"*70)
    
    all_results = TestResults()
    
    # Test configuration
    results = test_config()
    all_results.passed.extend(results.passed)
    all_results.failed.extend(results.failed)
    all_results.skipped.extend(results.skipped)
    
    # Get client for remaining tests
    try:
        client = get_client()
    except Exception as e:
        print(f"\n⚠ WARNING: Could not create client: {e}")
        client = None
    
    # Test word ladder (no client needed)
    results = test_word_ladder()
    all_results.passed.extend(results.passed)
    all_results.failed.extend(results.failed)
    all_results.skipped.extend(results.skipped)
    
    # Test GPT-2 CoT (no client needed)
    results = test_gpt2_cot()
    all_results.passed.extend(results.passed)
    all_results.failed.extend(results.failed)
    all_results.skipped.extend(results.skipped)
    
    # Test web search tool (no client needed)
    results = test_web_search_tool()
    all_results.passed.extend(results.passed)
    all_results.failed.extend(results.failed)
    all_results.skipped.extend(results.skipped)
    
    # Tests that require client (may be skipped if Ollama not running)
    if client:
        results = test_few_shot_cot(client)
        all_results.passed.extend(results.passed)
        all_results.failed.extend(results.failed)
        all_results.skipped.extend(results.skipped)
        
        results = test_zero_shot_cot(client)
        all_results.passed.extend(results.passed)
        all_results.failed.extend(results.failed)
        all_results.skipped.extend(results.skipped)
        
        results = test_self_consistency(client)
        all_results.passed.extend(results.passed)
        all_results.failed.extend(results.failed)
        all_results.skipped.extend(results.skipped)
        
        results = test_sequential_revision(client)
        all_results.passed.extend(results.passed)
        all_results.failed.extend(results.failed)
        all_results.skipped.extend(results.skipped)
        
        results = test_tree_of_thoughts(client)
        all_results.passed.extend(results.passed)
        all_results.failed.extend(results.failed)
        all_results.skipped.extend(results.skipped)
        
        results = test_simple_search_agent(client)
        all_results.passed.extend(results.passed)
        all_results.failed.extend(results.failed)
        all_results.skipped.extend(results.skipped)
    
    # Print summary
    success = all_results.summary()
    
    print("\n" + "#"*70)
    if success:
        print("# ALL TESTS COMPLETED SUCCESSFULLY!")
    else:
        print("# SOME TESTS FAILED - SEE SUMMARY ABOVE")
    print("#"*70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
