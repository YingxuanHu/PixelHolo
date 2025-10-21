"""AI interaction utilities including Ollama integration and internet search."""

import json
import random
import re
import shutil
import subprocess
import time
import urllib.parse
from typing import Dict, List, Optional

import cv2
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import (
    BASE_SYSTEM_PROMPT,
    DISPLAY_WINDOW_NAME,
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    THINKING_ICON_PATH,
)
from .timing import time_operation
from .ui import overlay_icon


def remove_emojis(text: str) -> str:
    """Remove emojis and non-ASCII characters from text."""
    emoji_pattern = re.compile(
        "["  # noqa: W605
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


internet_training_questions: List[str] = [
    "Who is the current president of the United States?",
    "What is the weather today?",
    "How tall is Mount Everest?",
    "Tell me about the latest iPhone release",
    "Show me the current Bitcoin price",
    "What's the score of the Lakers game tonight?",
    "How many people live in New York City?",
    "Explain quantum computing",
    "What's the capital of France?",
    "How do I bake a cake?",
    "Tell me about the history of the Roman Empire",
    "Who won the last Super Bowl?",
    "What movies are showing this weekend?",
    "What's the latest news about AI?",
    "How old is Elon Musk?",
    "What's the deadline for tax filing this year?",
    "How many calories are in a banana?",
    "Tell me a story about dragons",
    "What's the best way to learn Python?",
    "What should I do if I have a headache?",
    "Recommend a good Italian restaurant in Boston",
    "How do I change my car's oil?",
    "What's the recipe for lasagna?",
    "Give me tips for improving my memory",
    "What's the meaning of life?",
    "How do I fix a leaking faucet?",
    "What's the phone number for Domino's Pizza near me?",
    "Tell me about your hobbies?",
    "What's your favorite movie?",
    "What do you think about politics?",
    "Tell me about your childhood",
    "Do you like pizza?",
    "Tell me a joke",
    "What's 2 + 2?",
    "How do you feel today?",
    "Can you help me with my homework?",
    "Where were you born?",
    "What languages do you speak?",
    "Do you believe in ghosts?",
]

internet_training_labels: List[int] = [
    1,
    1,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(internet_training_questions)
ML_model = LogisticRegression()
ML_model.fit(X_train, internet_training_labels)


def needs_internet_ml(query: str) -> bool:
    """Determine if a query needs internet search using ML."""
    query_vec = vectorizer.transform([query])
    prediction = ML_model.predict(query_vec)[0]
    return prediction == 1


def web_search(query: str, max_results: int = 4) -> str:
    """Perform a Bing search and format the results."""
    try:
        time.sleep(random.uniform(1, 2))
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/search?q={encoded_query}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        snippets = []
        results_found = 0

        for result in soup.find_all("li", class_="b_algo"):
            if results_found >= max_results:
                break
            title_elem = result.find("h2")
            if title_elem:
                title_link = title_elem.find("a")
                if title_link:
                    title = title_link.get_text(strip=True)
                    href = title_link.get("href", "")
                else:
                    title = title_elem.get_text(strip=True)
                    href = ""
            else:
                title = ""
                href = ""

            body_elem = result.find("div", class_="b_caption")
            if body_elem:
                for unwanted in body_elem.find_all(["strong", "em"]):
                    unwanted.unwrap()
                body = body_elem.get_text(strip=True)
            else:
                body = ""

            if title and body:
                snippets.append(f"{title}\n{body}\n{href}\n")
                results_found += 1

        return "\n".join(snippets) if snippets else "No results found."
    except requests.exceptions.RequestException as exc:
        return f"Request failed: {exc}"
    except Exception as exc:
        return f"Search failed: {exc}"


def _ollama_cli_path() -> Optional[str]:
    """Return the Ollama executable path if available."""
    return shutil.which("ollama")


def _model_list_contains(model: str, listing: str) -> bool:
    """Check whether a model name appears in `ollama list` output."""
    target = model.split(":", 1)[0]
    for line in listing.splitlines():
        if not line.strip():
            continue
        name = line.split()[0]
        base = name.split(":", 1)[0]
        if name == model or base == target:
            return True
    return False


def _ensure_ollama_model(model: str) -> bool:
    """Ensure the requested Ollama model is installed, attempting a pull if needed."""
    executable = _ollama_cli_path()
    if executable is None:
        print("🔌 Ollama CLI not found on PATH; cannot auto-install models.")
        return False

    try:
        list_result = subprocess.run(
            [executable, "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if list_result.returncode == 0 and _model_list_contains(model, list_result.stdout):
            return True
    except Exception as exc:
        print(f"⚠️ Unable to inspect local Ollama models: {exc}")
        return False

    print(f"⬇️ Pulling Ollama model '{model}'...")
    try:
        with subprocess.Popen(
            [executable, "pull", model],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            assert proc.stdout
            for line in proc.stdout:
                if line:
                    print(line.rstrip())
            return_code = proc.wait()

        if return_code == 0:
            return True

        print(f"❌ Failed to pull model '{model}'. Exit code: {return_code}")
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if exc.stderr else str(exc)
        print(f"❌ Failed to pull model '{model}': {message}")
    except Exception as exc:  # pragma: no cover
        print(f"❌ Unexpected error pulling model '{model}': {exc}")
    return False


def _derive_chat_endpoint(url: str) -> Optional[str]:
    """Return the equivalent /api/chat endpoint for a given Ollama API URL."""
    if not url.endswith("/generate"):
        return None
    return url[: -len("/generate")] + "/chat"


def _extract_response_text(response_data: Dict[str, object]) -> Optional[str]:
    """Extract assistant text from either /api/generate or /api/chat responses."""
    candidate = response_data.get("response")
    if isinstance(candidate, str):
        return candidate.strip()
    message = response_data.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    return None


def get_ollama_response(
    prompt: str,
    system_prompt: Optional[str],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    base_img=None,
    internet_icon=None,
) -> str:
    """Send a prompt to the Ollama API and get a cleaned response."""
    search_context = ""
    if needs_internet_ml(prompt):
        print("🌐 Query requires internet search. Searching...")
        if base_img is not None and internet_icon is not None:
            img_with_internet = overlay_icon(base_img, internet_icon)
            cv2.imshow(DISPLAY_WINDOW_NAME, img_with_internet)
            cv2.waitKey(1)

        with time_operation("Web Search", verbose=True, track_memory=False):
            search_results = web_search(prompt)
        if search_results and "No results found" not in search_results:
            search_context = (
                f"\n\nCurrent search results for '{prompt}':\n{search_results}\n\n"
                "Please use this information to provide an accurate, up-to-date response."
            )
            print("✅ Search completed. Found relevant information.")
        else:
            print("❌ Search completed but no relevant results found.")

    print("🧠 Getting response from Ollama...")
    if base_img is not None:
        thinking_icon = cv2.imread(str(THINKING_ICON_PATH), cv2.IMREAD_UNCHANGED)
        if thinking_icon is not None:
            img_with_thinking = overlay_icon(base_img, thinking_icon)
            cv2.imshow(DISPLAY_WINDOW_NAME, img_with_thinking)
            cv2.waitKey(1)

    current_system_prompt = system_prompt or BASE_SYSTEM_PROMPT
    full_prompt = current_system_prompt + "\n\n"

    if conversation_history:
        for message in conversation_history[-10:]:
            role = message.get("role")
            content = message.get("content")
            if not role or not content:
                continue
            if role == "user":
                full_prompt += f"Human: {content}\n"
            else:
                full_prompt += f"Assistant: {content}\n"

    user_message = prompt + search_context
    full_prompt += f"Human: {user_message}\nAssistant: "

    messages: List[Dict[str, str]] = []
    if current_system_prompt:
        messages.append({"role": "system", "content": current_system_prompt})
    if conversation_history:
        for message in conversation_history[-10:]:
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant", "system"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 150},
    }
    headers = {"Content-Type": "application/json"}

    attempted_model_install = False
    attempted_chat_endpoint = False

    while True:
        try:
            with time_operation("Ollama API Call", verbose=True, track_memory=False):
                response = requests.post(OLLAMA_API_URL, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                response_data = response.json()

            extracted = _extract_response_text(response_data)
            if extracted:
                return remove_emojis(extracted)
            print("Unexpected response format from Ollama")
            break
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            body_text = exc.response.text if exc.response is not None else ""

            if status == 404 and not attempted_model_install:
                attempted_model_install = True
                if _ensure_ollama_model(OLLAMA_MODEL):
                    print("✅ Model downloaded. Retrying Ollama request...")
                    continue

            if status == 404 and not attempted_chat_endpoint:
                chat_url = _derive_chat_endpoint(OLLAMA_API_URL)
                if chat_url:
                    attempted_chat_endpoint = True
                    chat_payload = {
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": 150},
                    }
                    try:
                        chat_response = requests.post(chat_url, json=chat_payload, headers=headers, timeout=60)
                        chat_response.raise_for_status()
                        response_data = chat_response.json()
                        extracted = _extract_response_text(response_data)
                        if extracted:
                            print("ℹ️ Used Ollama chat endpoint fallback.")
                            return remove_emojis(extracted)
                    except requests.exceptions.RequestException as chat_exc:
                        print(f"Error connecting to Ollama chat endpoint: {chat_exc}")

            print(f"Error connecting to Ollama API: {exc}")
            if body_text:
                print(f"Ollama response body: {body_text.strip()}")
            print("Please ensure Ollama is running and the configured model exists.")
            break
        except requests.exceptions.RequestException as exc:
            print(f"Error connecting to Ollama API: {exc}")
            print("Please ensure Ollama is running on port 11434 with a model loaded.")
            break
        except json.JSONDecodeError as exc:
            print(f"Error parsing JSON response: {exc}")
            break
        except Exception as exc:
            print(f"Unexpected error: {exc}")
            break

    return "I seem to be having trouble thinking right now."
