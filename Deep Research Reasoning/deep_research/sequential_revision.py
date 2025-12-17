"""
Sequential revision module for iterative answer improvement.
Generates initial draft and refines it through multiple revision rounds.
"""

from typing import List
from openai import OpenAI


class SequentialRevision:
    """Sequential revision for iterative answer refinement."""
    
    def __init__(self, client: OpenAI, model: str = "llama3.2:3b"):
        """
        Initialize Sequential Revision.
        
        Args:
            client: OpenAI client instance
            model: Model name to use
        """
        self.client = client
        self.model = model
    
    def generate_initial_draft(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> str:
        """
        Generate the initial draft answer.
        
        Args:
            question: Question to answer
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Initial draft answer
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear, comprehensive answers."},
                {"role": "user", "content": question}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def revise_draft(
        self,
        question: str,
        previous_draft: str,
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> str:
        """
        Revise a previous draft to improve it.
        
        Args:
            question: Original question
            previous_draft: Previous answer draft
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Revised answer
        """
        revision_prompt = f"""Here is a previous answer to the question: "{question}"

Previous answer:
{previous_draft}

Please review and improve this answer by:
1. Adding more detail or clarity where needed
2. Correcting any errors or imprecisions
3. Making it more concise and well-structured

Provide an improved version:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that carefully revises and improves answers."},
                {"role": "user", "content": revision_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_with_revisions(
        self,
        question: str,
        max_steps: int = 3,
        temperature: float = 0.7,
        verbose: bool = True
    ) -> List[str]:
        """
        Generate an answer with multiple revision steps.
        
        Args:
            question: Question to answer
            max_steps: Total number of steps (initial + revisions)
            temperature: Sampling temperature
            verbose: If True, print each draft
            
        Returns:
            List of all drafts from initial to final
        """
        drafts = []
        
        # Generate initial draft
        if verbose:
            print("=" * 60)
            print("INITIAL DRAFT")
            print("=" * 60)
        
        draft = self.generate_initial_draft(question, temperature)
        drafts.append(draft)
        
        if verbose:
            print(draft)
            print()
        
        # Iteratively revise
        for step in range(1, max_steps):
            if verbose:
                print("=" * 60)
                print(f"REVISION {step}")
                print("=" * 60)
            
            draft = self.revise_draft(question, draft, temperature)
            drafts.append(draft)
            
            if verbose:
                print(draft)
                print()
        
        return drafts
    
    def get_final_answer(
        self,
        question: str,
        max_steps: int = 3,
        temperature: float = 0.7,
        verbose: bool = False
    ) -> str:
        """
        Get the final revised answer.
        
        Args:
            question: Question to answer
            max_steps: Total number of steps
            temperature: Sampling temperature
            verbose: If True, print each draft
            
        Returns:
            str: Final revised answer
        """
        drafts = self.generate_with_revisions(question, max_steps, temperature, verbose)
        return drafts[-1]


def sequential_revision(
    client: OpenAI,
    question: str,
    max_steps: int = 3,
    model: str = "llama3.2:3b",
    verbose: bool = True
) -> str:
    """
    Convenience function for sequential revision.
    
    Args:
        client: OpenAI client
        question: Question to answer
        max_steps: Number of revision steps
        model: Model name
        verbose: If True, print progress
        
    Returns:
        str: Final revised answer
    """
    sr = SequentialRevision(client, model)
    return sr.get_final_answer(question, max_steps, verbose=verbose)
