"""
Self-consistency module for improving reasoning accuracy.
Uses multiple sampling and majority voting to select the best answer.
"""

import re
import collections
from typing import List, Tuple, Dict
from openai import OpenAI


class SelfConsistency:
    """Self-consistency reasoning with majority voting."""
    
    def __init__(self, client: OpenAI, model: str = "llama3.2:3b"):
        """
        Initialize Self-Consistency.
        
        Args:
            client: OpenAI client instance
            model: Model name to use
        """
        self.client = client
        self.model = model
    
    def generate_single_answer(
        self,
        question: str,
        temperature: float = 1.0,
        max_tokens: int = 200
    ) -> str:
        """
        Generate a single CoT answer for a question.
        
        Args:
            question: Question to answer
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Extracted final answer
        """
        prompt = f"{question}\n\nLet's think step by step."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves problems step by step."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        text = response.choices[0].message.content
        return self._extract_answer(text)
    
    def generate_multiple(
        self,
        question: str,
        n: int = 10,
        temperature_range: Tuple[float, float] = (0.7, 1.0),
        verbose: bool = True
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate multiple reasoning chains and select winner by majority vote.
        
        Args:
            question: Question to answer
            n: Number of reasoning chains to generate
            temperature_range: (min, max) temperature values to vary
            verbose: If True, print each generated answer
            
        Returns:
            Tuple of (winning answer, vote counts dictionary)
        """
        answers = []
        temp_min, temp_max = temperature_range
        
        if verbose:
            print(f"Running {n} independent reasoning chains...\n")
        
        for i in range(n):
            # Vary temperature across runs for diversity
            temp = temp_min + (i % 3) * ((temp_max - temp_min) / 2)
            answer = self.generate_single_answer(question, temperature=temp)
            
            # Normalize answer for better matching
            normalized = answer.strip().lower()
            answers.append(normalized)
            
            if verbose:
                print(f"Run {i+1}: {normalized}")
        
        # Count occurrences and return most common
        counter = collections.Counter(answers)
        winner = counter.most_common(1)[0][0]
        
        return winner, dict(counter)
    
    @staticmethod
    def _extract_answer(text: str) -> str:
        """
        Extract the final answer from reasoning text.
        
        Args:
            text: Full reasoning text
            
        Returns:
            str: Extracted answer
        """
        # Look for common answer patterns
        patterns = [
            r'(?:answer|result|therefore)[:\s]+([^\n\.]+)',
            r'(\d+(?:\.\d+)?)\s*(?:$|\.)',
            r'the answer is[:\s]+([^\n\.]+)',
            r'final answer[:\s]+([^\n\.]+)'
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()
        
        # Fallback: return last sentence or last word
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            return sentences[-1]
        
        return text.split()[-1] if text.split() else text


def self_consistent_answer(
    client: OpenAI,
    question: str,
    n: int = 10,
    model: str = "llama3.2:3b",
    verbose: bool = True
) -> Tuple[str, Dict[str, int]]:
    """
    Convenience function for self-consistency reasoning.
    
    Args:
        client: OpenAI client
        question: Question to answer
        n: Number of reasoning chains
        model: Model name
        verbose: If True, print progress
        
    Returns:
        Tuple of (winning answer, vote counts)
    """
    sc = SelfConsistency(client, model)
    return sc.generate_multiple(question, n=n, verbose=verbose)
