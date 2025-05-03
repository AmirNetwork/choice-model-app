# services/deepseek.py
import subprocess
import re
import traceback

# For demonstration, we assume you run a local LLM with a command, e.g. "ollama run deepseek-r1:7b"
# Adjust the `LLM_COMMAND` or implement an API call depending on your real environment.

LLM_COMMAND = "ollama run deepseek-r1:7b"

def query_llm(prompt_text: str) -> str:
    """
    Sends a prompt to the local LLM via subprocess and returns the LLM's raw response.
    Modify if your environment differs (like calling an OpenAI or Hugging Face API).
    """
    try:
        process = subprocess.Popen(
            LLM_COMMAND,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            encoding='utf-8',
            errors='replace'
        )
        stdout, stderr = process.communicate(input=prompt_text + "\n", timeout=60)
        if process.returncode != 0:
            error_message = stderr.strip() or f"LLM command failed (code {process.returncode})"
            if "connection refused" in error_message.lower():
                return "Error: Cannot connect to LLM service."
            return f"Error calling LLM: {error_message}"

        response = stdout.strip()
        # Remove ANSI escape codes if any
        response = re.sub(r'\x1b\[.*?m', '', response)
        return response

    except subprocess.TimeoutExpired:
        return "Error: LLM request timed out. Please try again."
    except FileNotFoundError:
        return "Error: LLM command not found."
    except Exception as e:
        return f"Error running LLM: {e}\n{traceback.format_exc()}"

def query_model(prompt_text: str, purpose: str = "general") -> str:
    """
    High-level function to build prompts or do extra processing before calling `query_llm`.
    For more advanced usage, you might store or retrieve conversation context, etc.
    """
    # For demonstration, we simply pass the prompt text as-is, or do minimal customization
    if purpose == "classify_relevance":
        final_prompt = f'Is this query related to choice modeling? Answer "Yes" or "No".\nQuery: "{prompt_text}"'
    elif purpose == "classify_intent":
        final_prompt = (f'Does the user want to **build/apply a model using data** or ask a **general question**? '
                        f'Answer "Build Intent" or "General Question".\nQuery: "{prompt_text}"')
    else:
        final_prompt = prompt_text

    return query_llm(final_prompt)
