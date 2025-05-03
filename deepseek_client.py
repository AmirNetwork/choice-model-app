'''
deepseek_client.py

A self-contained client for invoking DeepSeek (via Ollama CLI) from Python.
Includes error handling, customizable model names, and timeout management.
Usage:
    from deepseek_client import DeepSeekClient
    client = DeepSeekClient(model="deepseek-r1:7b", ollama_path=r"C:\Program Files\Ollama", timeout=120)
    response = client.run("Your prompt here...")
    print(response)
'''
import os
import re
import subprocess
import traceback

class DeepSeekClient:
    def __init__(self, model: str = "deepseek-r1:7b", ollama_path: str = None, timeout: int = 60):
        """
        Initialize the DeepSeekClient.
        :param model: Ollama model identifier (e.g., "deepseek-r1:7b").
        :param ollama_path: Path to the Ollama executable folder. If None, assumes 'ollama' is on PATH.
        :param timeout: Request timeout in seconds.
        """
        self.model = model
        self.timeout = timeout
        self.env = os.environ.copy()
        if ollama_path:
            # prepend ollama_path so subprocess can find the 'ollama' command
            self.env["PATH"] = ollama_path + os.pathsep + self.env.get("PATH", "")

    def run(self, prompt_text: str) -> str:
        """
        Send the prompt to DeepSeek via Ollama CLI and return the text response.
        """
        command = ["ollama", "run", self.model]
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
                encoding='utf-8',
                errors='replace'
            )
            stdout, stderr = proc.communicate(input=prompt_text + "\n", timeout=self.timeout)
            if proc.returncode != 0:
                err = stderr.strip() or f"Ollama returned code {proc.returncode}"
                if "connection refused" in err.lower():
                    return "Error: Cannot connect to Ollama service."
                return f"Error calling DeepSeek: {err}"
            # strip color codes and ANSI escapes
            clean = re.sub(r'\x1b\[[0-9;]*m', '', stdout.strip())
            return clean

        except subprocess.TimeoutExpired:
            proc.kill()
            return f"Error: DeepSeek request timed out after {self.timeout} seconds."
        except FileNotFoundError:
            return ("Error: Ollama CLI not found. Ensure Ollama is installed and its path is correct. "
                    "You can specify ollama_path when creating DeepSeekClient.")
        except Exception as e:
            return f"Unexpected error: {e}\n{traceback.format_exc()}"


# Example usage:
if __name__ == "__main__":
    demo = DeepSeekClient(
        model="deepseek-r1:7b",
        ollama_path=r"C:\Program Files\Ollama",
        timeout=120
    )
    prompt = (
        "You are DeepSeek, an expert in Python.\n"
        "Write a function that reverses a string. Provide code in triple backticks."
    )
    answer = demo.run(prompt)
    print("DeepSeek Response:\n", answer)
