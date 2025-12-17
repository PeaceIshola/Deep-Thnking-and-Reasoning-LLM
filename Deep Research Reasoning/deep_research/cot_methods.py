"""
Chain-of-Thought (CoT) prompting methods.
Includes few-shot, zero-shot, and GPT-2 based CoT implementations.
"""

from openai import OpenAI
from typing import Optional, List, Dict
from transformers import pipeline
import torch


class FewShotCoT:
    """Few-shot Chain-of-Thought reasoning."""
    
    def __init__(self, client: OpenAI, model: str = "llama3.2:3b"):
        """
        Initialize Few-Shot CoT.
        
        Args:
            client: OpenAI client instance
            model: Model name to use
        """
        self.client = client
        self.model = model
    
    def create_prompt(self, examples: List[Dict[str, str]], question: str) -> str:
        """
        Create a few-shot prompt with examples and a new question.
        
        Args:
            examples: List of dicts with 'question' and 'answer' keys
            question: New question to answer
            
        Returns:
            str: Formatted prompt
        """
        prompt_parts = []
        
        for ex in examples:
            prompt_parts.append(f"Q: {ex['question']}")
            prompt_parts.append(f"A: {ex['answer']}\n")
        
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append("A: Let me think step by step:")
        
        return "\n".join(prompt_parts)
    
    def generate(
        self, 
        question: str, 
        examples: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> str:
        """
        Generate a few-shot CoT response.
        
        Args:
            question: Question to answer
            examples: List of example Q&A pairs (uses defaults if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Model's response
        """
        if examples is None:
            examples = self._get_default_examples()
        
        prompt = self.create_prompt(examples, question)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves problems step by step."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    @staticmethod
    def _get_default_examples() -> List[Dict[str, str]]:
        """Get default few-shot examples."""
        return [
            {
                "question": "If a train travels 60 miles in 1 hour, how far will it travel in 3.5 hours at the same speed?",
                "answer": """Let me think step by step:
1. The train's speed is 60 miles per hour
2. To find distance, I multiply speed × time
3. Distance = 60 miles/hour × 3.5 hours = 210 miles
Therefore, the train will travel 210 miles."""
            },
            {
                "question": "A store has 48 apples. If they sell 1/3 of them in the morning and 1/4 of the remaining in the afternoon, how many apples are left?",
                "answer": """Let me think step by step:
1. Morning sales: 1/3 of 48 = 48 ÷ 3 = 16 apples sold
2. Remaining after morning: 48 - 16 = 32 apples
3. Afternoon sales: 1/4 of 32 = 32 ÷ 4 = 8 apples sold
4. Final remaining: 32 - 8 = 24 apples
Therefore, 24 apples are left."""
            }
        ]


class ZeroShotCoT:
    """Zero-shot Chain-of-Thought reasoning."""
    
    def __init__(self, client: OpenAI, model: str = "llama3.2:3b"):
        """
        Initialize Zero-Shot CoT.
        
        Args:
            client: OpenAI client instance
            model: Model name to use
        """
        self.client = client
        self.model = model
    
    def generate(
        self,
        question: str,
        cot_trigger: str = "Let's think step by step.",
        temperature: float = 0.7,
        max_tokens: int = 300
    ) -> str:
        """
        Generate a zero-shot CoT response.
        
        Args:
            question: Question to answer
            cot_trigger: Phrase to trigger step-by-step reasoning
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            str: Model's response
        """
        prompt = f"{question}\n\n{cot_trigger}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that solves problems with clear step-by-step reasoning."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content


class GPT2CoT:
    """Chain-of-Thought with GPT-2 (non-instruction-tuned model)."""
    
    def __init__(self):
        """Initialize GPT-2 text generation pipeline."""
        self.generator = pipeline("text-generation", model="gpt2")
    
    def generate(
        self,
        question: str,
        examples: Optional[List[Dict[str, str]]] = None,
        decoding_strategy: str = "greedy",
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using GPT-2 with few-shot CoT.
        
        Args:
            question: Question to answer
            examples: Few-shot examples (uses defaults if None)
            decoding_strategy: 'greedy', 'top_k', or 'nucleus'
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            str: Generated text
        """
        if examples is None:
            examples = self._get_default_examples()
        
        prompt = self._create_prompt(examples, question)
        
        # Configure decoding based on strategy
        if decoding_strategy == "greedy":
            output = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_new_tokens,
                do_sample=False,
                pad_token_id=50256
            )
        elif decoding_strategy == "top_k":
            output = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_new_tokens,
                do_sample=True,
                top_k=top_k,
                temperature=temperature,
                pad_token_id=50256
            )
        else:  # nucleus
            output = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=50256
            )
        
        # Extract only the newly generated text
        generated = output[0]['generated_text'][len(prompt):]
        return generated
    
    @staticmethod
    def _get_default_examples() -> List[Dict[str, str]]:
        """Get default few-shot examples formatted for GPT-2."""
        return [
            {
                "question": "If I have 3 apples and buy 2 more, how many do I have?",
                "answer": "Step 1: Start with 3 apples. Step 2: Add 2 more apples. Step 3: 3 + 2 = 5. Answer: 5 apples."
            },
            {
                "question": "A rectangle has length 6 and width 4. What is its area?",
                "answer": "Step 1: Area formula is length × width. Step 2: 6 × 4 = 24. Answer: 24."
            }
        ]
    
    @staticmethod
    def _create_prompt(examples: List[Dict[str, str]], question: str) -> str:
        """Create GPT-2 style prompt."""
        prompt_parts = []
        
        for ex in examples:
            prompt_parts.append(f"Q: {ex['question']}")
            prompt_parts.append(f"A: {ex['answer']}\n")
        
        prompt_parts.append(f"Q: {question}")
        prompt_parts.append("A:")
        
        return "\n".join(prompt_parts)


# Convenience functions
def few_shot_cot(
    client: OpenAI,
    question: str,
    examples: Optional[List[Dict[str, str]]] = None,
    model: str = "llama3.2:3b",
    **kwargs
) -> str:
    """
    Convenience function for few-shot CoT.
    
    Args:
        client: OpenAI client
        question: Question to answer
        examples: Optional few-shot examples
        model: Model name
        **kwargs: Additional arguments passed to generate()
        
    Returns:
        str: Generated response
    """
    cot = FewShotCoT(client, model)
    return cot.generate(question, examples, **kwargs)


def zero_shot_cot(
    client: OpenAI,
    question: str,
    model: str = "llama3.2:3b",
    **kwargs
) -> str:
    """
    Convenience function for zero-shot CoT.
    
    Args:
        client: OpenAI client
        question: Question to answer
        model: Model name
        **kwargs: Additional arguments passed to generate()
        
    Returns:
        str: Generated response
    """
    cot = ZeroShotCoT(client, model)
    return cot.generate(question, **kwargs)
