"""
Tree-of-Thoughts (ToT) module for exploring multiple reasoning paths.
Includes word ladder puzzle solver and generic ToT search.
"""

import re
from typing import List, Tuple, Set, Optional, Callable
from openai import OpenAI


class WordLadder:
    """Word ladder puzzle solver using tree-of-thoughts approach."""
    
    @staticmethod
    def neighbors(word: str, vocabulary: Set[str]) -> List[str]:
        """
        Generate all valid one-letter mutations of a word.
        
        Args:
            word: Current word
            vocabulary: Set of valid words
            
        Returns:
            List of valid neighboring words
        """
        results = []
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != word[i]:
                    candidate = word[:i] + c + word[i+1:]
                    if candidate in vocabulary:
                        results.append(candidate)
        return results
    
    @staticmethod
    def edit_distance(word1: str, word2: str) -> int:
        """
        Calculate Levenshtein distance between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            int: Edit distance
        """
        if len(word1) != len(word2):
            return sum(1 for a, b in zip(word1, word2) if a != b) + abs(len(word1) - len(word2))
        return sum(1 for a, b in zip(word1, word2) if a != b)
    
    def solve(
        self,
        start: str,
        goal: str,
        vocabulary: Set[str],
        max_depth: int = 5,
        beam_width: int = 4
    ) -> Optional[List[str]]:
        """
        Solve word ladder puzzle using beam search.
        
        Args:
            start: Starting word
            goal: Goal word
            vocabulary: Set of valid words
            max_depth: Maximum search depth
            beam_width: Number of paths to keep at each step
            
        Returns:
            List of words forming solution path, or None if not found
        """
        # Initialize frontier with single path
        frontier = [[start]]
        visited = {start}
        
        # Beam search
        for depth in range(max_depth):
            candidates = []
            
            # Expand each path
            for path in frontier:
                last_word = path[-1]
                
                # Check if goal reached
                if last_word == goal:
                    return path
                
                # Generate neighbors
                for neighbor in self.neighbors(last_word, vocabulary):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        score = self.edit_distance(neighbor, goal)
                        candidates.append((score, new_path))
                        visited.add(neighbor)
            
            if not candidates:
                break
            
            # Keep top beam_width paths
            candidates.sort(key=lambda x: x[0])
            frontier = [path for score, path in candidates[:beam_width]]
        
        # Check if any frontier path reached goal
        for path in frontier:
            if path[-1] == goal:
                return path
        
        return None


class GenericToT:
    """Generic Tree-of-Thoughts search for reasoning problems."""
    
    def __init__(self, client: OpenAI, model: str = "llama3.2:3b"):
        """
        Initialize Generic ToT.
        
        Args:
            client: OpenAI client instance
            model: Model name to use
        """
        self.client = client
        self.model = model
    
    def propose_thoughts(
        self,
        question: str,
        state: str,
        k: int = 2,
        temperature: float = 0.8
    ) -> List[str]:
        """
        Propose k next reasoning steps.
        
        Args:
            question: Original question
            state: Current partial solution
            k: Number of thoughts to generate
            temperature: Sampling temperature
            
        Returns:
            List of proposed next steps
        """
        if state:
            prompt = f"Problem: {question}\n\nCurrent partial solution:\n{state}\n\nPropose the next step to continue this solution. Be brief and specific:"
        else:
            prompt = f"Problem: {question}\n\nPropose the first step toward solving this. Be brief and specific:"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=100,
            n=k
        )
        
        thoughts = [choice.message.content.strip() for choice in response.choices]
        return thoughts[:k]
    
    def score_state(
        self,
        question: str,
        state: str,
        temperature: float = 0.3
    ) -> int:
        """
        Score how promising a partial solution is (1-10 scale).
        
        Args:
            question: Original question
            state: Current partial solution
            temperature: Sampling temperature
            
        Returns:
            int: Score from 1-10
        """
        prompt = f"""Problem: {question}

Partial solution:
{state}

Rate how promising this partial solution is on a scale of 1-10, where:
- 1-3: Poor approach, unlikely to lead to a good solution
- 4-6: Okay approach, has some potential
- 7-9: Good approach, likely to lead to a solid solution
- 10: Excellent approach, very promising

Respond with just a number from 1 to 10:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=10
        )
        
        text = response.choices[0].message.content.strip()
        # Parse first integer between 1-10
        match = re.search(r'\b([1-9]|10)\b', text)
        if match:
            return int(match.group(1))
        return 5  # Default middle score
    
    def search(
        self,
        question: str,
        depth: int = 2,
        width: int = 2,
        verbose: bool = True
    ) -> Tuple[str, int]:
        """
        Run tree-of-thoughts search.
        
        Args:
            question: Question to solve
            depth: Search depth
            width: Beam width (branches per node)
            verbose: If True, print search progress
            
        Returns:
            Tuple of (best solution state, best score)
        """
        # Initialize with empty state
        frontier = [("", 0)]
        
        if verbose:
            print(f"=== Tree-of-Thoughts Search (depth={depth}, width={width}) ===\n")
        
        for d in range(depth):
            if verbose:
                print(f"--- Depth {d+1} ---")
            
            candidates = []
            
            # Expand each state
            for state, prev_score in frontier:
                thoughts = self.propose_thoughts(question, state, k=width)
                
                for thought in thoughts:
                    # Build new state
                    if state:
                        new_state = f"{state}\n\nStep {d+2}: {thought}"
                    else:
                        new_state = f"Step 1: {thought}"
                    
                    # Score new state
                    score = self.score_state(question, new_state)
                    candidates.append((new_state, score))
                    
                    if verbose:
                        print(f"  Score {score}: {thought[:60]}...")
            
            # Keep top width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            frontier = candidates[:width]
            
            if verbose:
                print()
        
        # Return best solution
        best_state, best_score = frontier[0]
        return best_state, best_score


# Convenience functions
def solve_word_ladder(
    start: str,
    goal: str,
    vocabulary: Set[str],
    max_depth: int = 5,
    beam_width: int = 4
) -> Optional[List[str]]:
    """
    Convenience function to solve word ladder puzzle.
    
    Args:
        start: Starting word
        goal: Goal word
        vocabulary: Set of valid words
        max_depth: Maximum search depth
        beam_width: Beam width for search
        
    Returns:
        Solution path or None
    """
    ladder = WordLadder()
    return ladder.solve(start, goal, vocabulary, max_depth, beam_width)


def tree_of_thoughts_search(
    client: OpenAI,
    question: str,
    depth: int = 2,
    width: int = 2,
    model: str = "llama3.2:3b",
    verbose: bool = True
) -> Tuple[str, int]:
    """
    Convenience function for tree-of-thoughts search.
    
    Args:
        client: OpenAI client
        question: Question to solve
        depth: Search depth
        width: Beam width
        model: Model name
        verbose: If True, print progress
        
    Returns:
        Tuple of (best solution, score)
    """
    tot = GenericToT(client, model)
    return tot.search(question, depth, width, verbose)
