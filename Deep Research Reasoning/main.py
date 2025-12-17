"""
Main entry point demonstrating all deep research system modules.
Run different reasoning methods independently or together.
"""

import argparse
from deep_research.config import get_client, Config
from deep_research.cot_methods import FewShotCoT, ZeroShotCoT, GPT2CoT
from deep_research.self_consistency import self_consistent_answer
from deep_research.sequential_revision import sequential_revision
from deep_research.tree_of_thoughts import solve_word_ladder, tree_of_thoughts_search
from deep_research.deep_research_agent import create_research_agent, SimpleSearchAgent


def demo_few_shot_cot(client):
    """Demonstrate few-shot Chain-of-Thought."""
    print("\n" + "="*70)
    print("FEW-SHOT CHAIN-OF-THOUGHT DEMO")
    print("="*70)
    
    cot = FewShotCoT(client, Config.DEFAULT_BASE_MODEL)
    question = "If 5 machines can produce 100 widgets in 2 hours, how many widgets can 8 machines produce in 3 hours?"
    
    print(f"\nQuestion: {question}\n")
    answer = cot.generate(question)
    print(f"Answer:\n{answer}")


def demo_zero_shot_cot(client):
    """Demonstrate zero-shot Chain-of-Thought."""
    print("\n" + "="*70)
    print("ZERO-SHOT CHAIN-OF-THOUGHT DEMO")
    print("="*70)
    
    cot = ZeroShotCoT(client, Config.DEFAULT_BASE_MODEL)
    question = "If a car travels at 45 mph for 2 hours, then 60 mph for 3 hours, what is the total distance?"
    
    print(f"\nQuestion: {question}\n")
    answer = cot.generate(question)
    print(f"Answer:\n{answer}")


def demo_gpt2_cot():
    """Demonstrate GPT-2 Chain-of-Thought."""
    print("\n" + "="*70)
    print("GPT-2 CHAIN-OF-THOUGHT DEMO")
    print("="*70)
    
    cot = GPT2CoT()
    question = "If a book costs $12 and I have $50, how much change will I get?"
    
    print(f"\nQuestion: {question}\n")
    
    # Try different decoding strategies
    for strategy in ["greedy", "top_k", "nucleus"]:
        print(f"\n--- {strategy.upper()} DECODING ---")
        answer = cot.generate(question, decoding_strategy=strategy, max_new_tokens=40)
        print(answer)


def demo_self_consistency(client):
    """Demonstrate self-consistency."""
    print("\n" + "="*70)
    print("SELF-CONSISTENCY DEMO")
    print("="*70)
    
    question = "What is the square root of 144?"
    print(f"\nQuestion: {question}\n")
    
    winner, votes = self_consistent_answer(client, question, n=5, model=Config.DEFAULT_BASE_MODEL)
    
    print("\n" + "="*50)
    print(f"Vote Distribution: {votes}")
    print(f"Winning Answer: {winner}")


def demo_sequential_revision(client):
    """Demonstrate sequential revision."""
    print("\n" + "="*70)
    print("SEQUENTIAL REVISION DEMO")
    print("="*70)
    
    question = "Explain how photosynthesis works and why it's important for life on Earth."
    print(f"\nQuestion: {question}\n")
    
    final_answer = sequential_revision(
        client, 
        question, 
        max_steps=3, 
        model=Config.DEFAULT_BASE_MODEL,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("FINAL REVISED ANSWER")
    print("="*60)
    print(final_answer)


def demo_word_ladder():
    """Demonstrate word ladder puzzle solver."""
    print("\n" + "="*70)
    print("WORD LADDER DEMO")
    print("="*70)
    
    vocab = {"hit", "dot", "cog", "log", "dog", "lot", "lit", "hot"}
    start = "hit"
    goal = "cog"
    
    print(f"\nSolving word ladder from '{start}' to '{goal}'")
    print(f"Vocabulary: {vocab}\n")
    
    result = solve_word_ladder(start, goal, vocab)
    
    if result:
        print(f"Solution path: {' -> '.join(result)}")
    else:
        print("No solution found")


def demo_tree_of_thoughts(client):
    """Demonstrate generic tree-of-thoughts search."""
    print("\n" + "="*70)
    print("TREE-OF-THOUGHTS DEMO")
    print("="*70)
    
    question = "Design a plan for a weekend science workshop for 12-year-olds."
    print(f"\nQuestion: {question}\n")
    
    solution, score = tree_of_thoughts_search(
        client,
        question,
        depth=2,
        width=2,
        model=Config.DEFAULT_BASE_MODEL,
        verbose=True
    )
    
    print(f"\n{'='*60}")
    print(f"BEST SOLUTION (Score: {score})")
    print("="*60)
    print(solution)


def demo_deep_research():
    """Demonstrate deep research agent."""
    print("\n" + "="*70)
    print("DEEP RESEARCH AGENT DEMO")
    print("="*70)
    
    question = "What are the best resources to learn machine learning in 2025?"
    print(f"\nQuestion: {question}\n")
    
    try:
        agent = create_research_agent(Config.DEFAULT_REASONING_MODEL)
        answer = agent.research(question)
        
        print(f"\n{'='*60}")
        print("RESEARCH ANSWER")
        print("="*60)
        print(answer)
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure Ollama is running and the model is installed")


def demo_simple_search_agent(client):
    """Demonstrate simple search agent with manual control."""
    print("\n" + "="*70)
    print("SIMPLE SEARCH AGENT DEMO")
    print("="*70)
    
    question = "What are the latest developments in quantum computing?"
    print(f"\nQuestion: {question}\n")
    
    try:
        agent = SimpleSearchAgent(client, Config.DEFAULT_REASONING_MODEL, max_iterations=3)
        answer = agent.research(question, verbose=True)
        
        print(f"\n{'='*60}")
        print("FINAL RESULT")
        print("="*60)
        print(answer)
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure Ollama is running and the model is installed")


def run_all_demos(client):
    """Run all demonstrations."""
    print("\n" + "#"*70)
    print("# DEEP RESEARCH REASONING SYSTEM - FULL DEMO")
    print("#"*70)
    
    try:
        demo_few_shot_cot(client)
    except Exception as e:
        print(f"Few-shot CoT demo failed: {e}")
    
    try:
        demo_zero_shot_cot(client)
    except Exception as e:
        print(f"Zero-shot CoT demo failed: {e}")
    
    try:
        demo_gpt2_cot()
    except Exception as e:
        print(f"GPT-2 CoT demo failed: {e}")
    
    try:
        demo_self_consistency(client)
    except Exception as e:
        print(f"Self-consistency demo failed: {e}")
    
    try:
        demo_sequential_revision(client)
    except Exception as e:
        print(f"Sequential revision demo failed: {e}")
    
    try:
        demo_word_ladder()
    except Exception as e:
        print(f"Word ladder demo failed: {e}")
    
    try:
        demo_tree_of_thoughts(client)
    except Exception as e:
        print(f"Tree-of-thoughts demo failed: {e}")
    
    try:
        demo_deep_research()
    except Exception as e:
        print(f"Deep research demo failed: {e}")
    
    print("\n" + "#"*70)
    print("# ALL DEMOS COMPLETED")
    print("#"*70)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(description="Deep Research Reasoning System")
    parser.add_argument(
        "--demo",
        choices=[
            "few-shot", "zero-shot", "gpt2", "self-consistency",
            "sequential", "word-ladder", "tree-of-thoughts",
            "research", "simple-search", "all"
        ],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API instead of Ollama"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (if using OpenAI)"
    )
    
    args = parser.parse_args()
    
    # Get client
    client = get_client(use_openai=args.use_openai, api_key=args.api_key)
    
    # Run selected demo
    demos = {
        "few-shot": demo_few_shot_cot,
        "zero-shot": demo_zero_shot_cot,
        "gpt2": demo_gpt2_cot,
        "self-consistency": demo_self_consistency,
        "sequential": demo_sequential_revision,
        "word-ladder": demo_word_ladder,
        "tree-of-thoughts": demo_tree_of_thoughts,
        "research": demo_deep_research,
        "simple-search": demo_simple_search_agent,
    }
    
    if args.demo == "all":
        run_all_demos(client)
    elif args.demo in ["gpt2", "word-ladder", "research"]:
        # These don't need client
        if args.demo == "gpt2":
            demo_gpt2_cot()
        elif args.demo == "word-ladder":
            demo_word_ladder()
        else:
            demo_deep_research()
    else:
        demos[args.demo](client)


if __name__ == "__main__":
    main()
