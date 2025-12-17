# Deep Research Reasoning System

Advanced LLM reasoning techniques for better problem-solving: Chain-of-Thought, Self-Consistency, Sequential Revision, Tree-of-Thoughts, and Web-Integrated Research Agents.

## Features

- **Chain-of-Thought**: Step-by-step reasoning (few-shot, zero-shot, GPT-2)
- **Self-Consistency**: Multiple answers with voting
- **Sequential Revision**: Iterative answer refinement
- **Tree-of-Thoughts**: Multi-path exploration
- **Research Agent**: Web search + reasoning

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test_deep_research.py

# Try demos
python main.py --demo word-ladder  # No API needed
python main.py --demo gpt2         # Uses local GPT-2
python main.py --demo all          # Requires Ollama
```

## Requirements

- **Python 3.9+**
- **Ollama** (for reasoning models) or **OpenAI API**
- Install models: `ollama pull llama3.2:3b` and `ollama pull deepseek-r1:8b`

## Project Structure

```
Deep Research Reasoning/
├── deep_research/              # Main package
│   ├── config.py              # Client setup
│   ├── cot_methods.py         # Chain-of-Thought
│   ├── self_consistency.py    # Voting-based reasoning
│   ├── sequential_revision.py # Iterative refinement
│   ├── tree_of_thoughts.py    # Multi-path search
│   └── deep_research_agent.py # Web + reasoning
├── main.py                    # Demo runner
├── test_deep_research.py      # Test suite
└── requirements.txt           # Dependencies
```

## Usage

### Command Line

```bash
# Run specific demos
python main.py --demo word-ladder      # Word puzzle solver
python main.py --demo gpt2             # GPT-2 Chain-of-Thought
python main.py --demo few-shot         # Few-shot CoT
python main.py --demo zero-shot        # Zero-shot CoT
python main.py --demo self-consistency # Voting mechanism
python main.py --demo sequential       # Answer refinement
python main.py --demo tree-of-thoughts # Multi-path search
python main.py --demo simple-search    # Web research agent
python main.py --demo all              # Run everything

# Use OpenAI instead of Ollama
python main.py --demo few-shot --use-openai --api-key YOUR_KEY
```

### Python API

```python
from deep_research.config import get_client
from deep_research.cot_methods import FewShotCoT, ZeroShotCoT
from deep_research.self_consistency import self_consistent_answer
from deep_research.sequential_revision import sequential_revision
from deep_research.tree_of_thoughts import solve_word_ladder, tree_of_thoughts_search
from deep_research.deep_research_agent import SimpleSearchAgent

client = get_client()  # Uses Ollama by default

# Chain-of-Thought
cot = FewShotCoT(client, "llama3.2:3b")
answer = cot.generate("If 5 machines produce 100 widgets in 2 hours, how many can 8 produce in 3?")

# Self-Consistency (vote on best answer)
winner, votes = self_consistent_answer(client, "What is the square root of 144?", n=5)

# Sequential Revision (iterative refinement)
final = sequential_revision(client, "Explain photosynthesis", max_steps=3)

# Tree-of-Thoughts (explore multiple paths)
solution, score = tree_of_thoughts_search(client, "Design a science workshop", depth=2, width=2)

# Word Ladder puzzle
path = solve_word_ladder("hit", "cog", {"hit", "hot", "dot", "dog", "cog"})

# Web Research
agent = SimpleSearchAgent(client, "deepseek-r1:8b", max_iterations=3)
answer = agent.research("What are the latest ML trends?")
```

## Testing

All modules are tested in [test_deep_research.py](Deep%20Research%20Reasoning/test_deep_research.py):

```bash
python test_deep_research.py
```

**Test Coverage:**
- ✓ Configuration and client setup
- ✓ Word Ladder solver
- ✓ GPT-2 Chain-of-Thought (greedy, top-k, nucleus)
- ✓ Web Search Tool
- ✓ Few-Shot CoT (with Ollama)
- ✓ Zero-Shot CoT (with Ollama)
- ✓ Self-Consistency voting
- ✓ Sequential Revision
- ✓ Tree-of-Thoughts search
- ✓ Simple Search Agent

## Key Methods Explained

**Chain-of-Thought**: Prompt model to show reasoning steps before answering

**Self-Consistency**: Generate N answers, pick most common (majority vote)

**Sequential Revision**: Generate answer → critique → improve → repeat

**Tree-of-Thoughts**: Breadth-first search through reasoning space, score each path

**Research Agent**: Use web search as a tool, let model decide when to search

## Requirements Details

```txt
openai>=1.0.0           # API client
transformers>=4.30.0    # HuggingFace models
torch>=2.0.0            # PyTorch
langchain>=0.1.0        # Agent framework
duckduckgo-search>=4.0.0 # Web search
numpy>=1.24.0           # Numerical ops
```

## Troubleshooting

**"Connection refused"**: Start Ollama with `ollama serve`

**"Model not found"**: Install models: `ollama pull llama3.2:3b`

**Import errors**: Install dependencies: `pip install -r requirements.txt`

**Slow performance**: Use smaller models or reduce iterations (n, max_steps, depth)

## License

MIT License - See LICENSE file

---

**Quick Reference:**
- Test everything: `python test_deep_research.py`
- Run demos: `python main.py --demo all`
- Import modules: `from deep_research.config import get_client`
