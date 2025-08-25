#!/usr/bin/env python3
"""
Script to generate Methods section for scientific manuscripts from Jupytext files.
Uses litemind to interface with LLM providers.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from litemind import OpenAIApi
    from litemind.agent.agent import Agent
    from litemind.agent.messages.message import Message
    # Only import APIs that are available - we'll handle others dynamically
    AnthropicApi = None
    GeminiApi = None
    combinedAPI = None
    try:
        from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi
    except ImportError:
        pass
    try:
        from litemind.apis.providers.google.google_api import GeminiApi
    except ImportError:
        pass
    try:
        from litemind import combinedAPI
    except ImportError:
        pass
except ImportError:
    print("Error: litemind is required but not installed. Install with: pip install litemind")
    sys.exit(1)


# Journal-specific system prompts
JOURNAL_PROMPTS = {
    "nature": """You are a scientific writing assistant specializing in Nature journal style. 
Generate a concise, precise Methods section from the provided Jupytext analysis code.
Nature Methods sections are typically brief but comprehensive, focusing on:
- Clear step-by-step procedures
- Statistical methods and software versions
- Parameter settings and thresholds
- Data processing workflows
- Justification for methodological choices

Write in past tense, third person. Be specific about computational tools, versions, and parameters.
Keep descriptions concise but complete. Include relevant citations where appropriate (as [ref]).
""",

    "cell": """You are a scientific writing assistant specializing in Cell journal style.
Generate a detailed Methods section from the provided Jupytext analysis code.
Cell Methods sections should be comprehensive and include:
- Detailed computational procedures
- Software tools with version numbers
- Statistical approaches and rationale
- Data preprocessing steps
- Quality control measures
- Parameter optimization rationale

Write in past tense, active voice where possible. Provide sufficient detail for reproducibility.
Include subsection headers where appropriate.
""",

    "science": """You are a scientific writing assistant specializing in Science journal style.
Generate a clear, methodical Methods section from the provided Jupytext analysis code.
Science Methods sections emphasize:
- Clear experimental design
- Computational workflow logic
- Statistical validation approaches
- Tool selection justification
- Reproducibility considerations

Write concisely but with sufficient technical detail. Use past tense, third person.
Focus on the logical flow of the analysis pipeline.
""",

    "generic": """You are a scientific writing assistant for academic manuscripts.
Generate a Methods section from the provided Jupytext analysis code.
The Methods section should include:
- Step-by-step computational procedures
- Software tools and versions used
- Statistical methods and parameters
- Data processing pipeline
- Quality control measures
- Rationale for methodological choices

Write in academic style, past tense, third person. Be precise and comprehensive
enough for reproducibility while maintaining clarity.
"""
}

def read_jupytext_file(filepath: str) -> str:
    """Read and return contents of a Jupytext .py file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error reading file {filepath}: {str(e)}")

def extract_analysis_content(jupytext_content: str) -> str:
    """
    Extract relevant analysis content from Jupytext file.
    Focus on markdown cells and key code sections with comments.
    """
    lines = jupytext_content.split('\n')
    relevant_content = []
    in_markdown = False
    
    for line in lines:
        # Detect markdown cells
        if line.strip().startswith('# %%') and '[markdown]' in line:
            in_markdown = True
            continue
        elif line.strip().startswith('# %%') and '[markdown]' not in line:
            in_markdown = False
            continue
        
        # Include markdown content
        if in_markdown:
            if line.startswith('# '):
                # Remove the '# ' prefix from markdown
                relevant_content.append(line[2:])
            else:
                relevant_content.append(line)
        # Include commented code that might contain methodology explanations
        elif line.strip().startswith('#') and len(line.strip()) > 1:
            relevant_content.append(line)
        # Include import statements and key function definitions
        elif any(keyword in line for keyword in ['import ', 'from ', 'def ', 'class ']):
            relevant_content.append(line)
    
    return '\n'.join(relevant_content)

def get_api_and_model(model: str) -> tuple:
    """Get appropriate API instance and model name based on model specification."""
    
    # Map model families to their API providers
    if model.startswith('gpt-') or model.startswith('o1-') or model.startswith('o3-') or model.startswith('chatgpt-'):
        # OpenAI models
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise Exception("OPENAI_API_KEY environment variable not set for OpenAI models")
        return OpenAIApi(), model
        
    elif model.startswith('claude-') or 'sonnet' in model.lower() or 'haiku' in model.lower() or 'opus' in model.lower():
        # Anthropic Claude models
        if AnthropicApi is None:
            raise Exception("AnthropicApi not available. Please check your litemind installation.")
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise Exception("ANTHROPIC_API_KEY environment variable not set for Claude models")
        
        # Map common Claude model names to official names
        claude_models = {
            'claude-sonnet-4': 'claude-3-sonnet-20240229',  # Update when sonnet-4 is available
            'claude-sonnet-3.5': 'claude-3-5-sonnet-20241022',
            'claude-haiku': 'claude-3-haiku-20240307',
            'claude-opus': 'claude-3-opus-20240229'
        }
        model_name = claude_models.get(model, model)
        return AnthropicApi(), model_name
        
    elif model.startswith('gemini-') or model.startswith('models/gemini'):
        # Google Gemini models
        if GeminiApi is None:
            raise Exception("GeminiApi not available. Please check your litemind installation.")
        api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise Exception("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set for Gemini models")
        
        # Ensure proper model format
        if not model.startswith('models/'):
            model = f'models/{model}'
        return GeminiApi(), model
        
    elif model == 'auto' or model == 'combined':
        # Use combinedAPI for automatic model selection
        if combinedAPI is None:
            raise Exception("combinedAPI not available. Please check your litemind installation.")
        return combinedAPI(), None
        
    else:
        # Default to OpenAI for unknown models
        print(f"Warning: Unknown model '{model}', defaulting to OpenAI API")
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise Exception("OPENAI_API_KEY environment variable not set")
        return OpenAIApi(), model

def generate_methods_section(
    jupytext_content: str, 
    journal_style: str = "generic",
    model: str = "o3-high",
    additional_context: Optional[str] = None
) -> str:
    """Generate Methods section using litemind."""
    
    # Get appropriate API and model
    api, model_name = get_api_and_model(model)
    
    # Get appropriate system prompt
    system_prompt = JOURNAL_PROMPTS.get(journal_style, JOURNAL_PROMPTS["generic"])
    
    # Extract relevant analysis content
    analysis_content = extract_analysis_content(jupytext_content)
    
    # Construct user prompt
    user_prompt = f"""Based on the following Jupytext analysis code, generate a Methods section for a scientific manuscript:

```python
{analysis_content}
```
"""
    
    if additional_context:
        user_prompt += f"\n\nAdditional context: {additional_context}"
    
    user_prompt += """

Please generate a well-structured Methods section that describes:
1. The computational approach and workflow
2. Software tools and versions used
3. Statistical methods and parameters
4. Data processing steps
5. Quality control measures
6. Justification for methodological choices

Format the output in markdown with appropriate subsections if needed."""

    try:
        # Create the agent
        agent = Agent(api=api)
        
        # Add system message
        agent.append_system_message(system_prompt)
        
        # Create message
        message = Message()
        message.append_text(user_prompt)
        
        # Use agent interface (this is the preferred method for litemind)
        response = agent(message)
        return str(response)
    
    except Exception as e:
        raise Exception(f"Error generating methods section: {str(e)}")

def save_output(content: str, output_path: str) -> None:
    """Save the generated methods section to a markdown file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Methods section saved to: {output_path}")
    except Exception as e:
        raise Exception(f"Error saving output to {output_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate Methods section for scientific manuscripts from Jupytext files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python draft_method_from_jupytext.py notebook.py
  python draft_method_from_jupytext.py notebook.py --journal-style nature --model claude-sonnet-4
  python draft_method_from_jupytext.py notebook.py --model gpt-4o --output methods.md
  python draft_method_from_jupytext.py notebook.py --context "Single-cell RNA-seq analysis" --model auto
        """
    )
    
    parser.add_argument(
        "input_file", 
        help="Path to Jupytext (.py) file containing analysis workflow"
    )
    
    parser.add_argument(
        "--journal-style", 
        choices=["nature", "cell", "science", "generic"],
        default="generic",
        help="Journal style for Methods section formatting (default: generic)"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="o3-high",
        help="""Model to use for generation. Options include:
        OpenAI: o3-high (default), o3-medium, o3-low, gpt-4o, gpt-4, etc.
        Claude: claude-sonnet-4, claude-sonnet-3.5, claude-haiku, claude-opus
        Gemini: gemini-1.5-pro, gemini-1.0-pro
        Auto: auto (automatic model selection)"""
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output markdown file path (default: <input_basename>_methods.md)"
    )
    
    parser.add_argument(
        "--context",
        help="Additional context about the analysis to include in the prompt"
    )
    

    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Set output file path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"{input_path.stem}_methods.md"
    
    try:
        # Read input file
        print(f"Reading Jupytext file: {args.input_file}")
        jupytext_content = read_jupytext_file(args.input_file)
        
        # Generate methods section
        print(f"Generating Methods section using {args.model} in {args.journal_style} style...")
        methods_content = generate_methods_section(
            jupytext_content=jupytext_content,
            journal_style=args.journal_style,
            model=args.model,
            additional_context=args.context
        )
        
        # Save output
        save_output(methods_content, str(output_path))
        
        print("✓ Methods section generated successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()