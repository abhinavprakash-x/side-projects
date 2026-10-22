# import requests

# url = "http://localhost:11434/api/chat"

# payload = {
#     "model": "qwen3.5:9b",
#     "messages": [
#         {
#             "role": "system",
#             "content": "You are a concise systems programming tutor."
#         },
#         {
#             "role": "user",
#             "content": "Explain a page fault in an operating system."
#         }
#     ],
#     "stream": False
# }

# try:
#     response = requests.post(url, json=payload, timeout=180)
#     response.raise_for_status()

#     data = response.json()
#     print(data["message"]["content"])

# except requests.RequestException as error:
#     print(f"Request failed: {error}")



import json
import sys
import requests


def ask_ollama(prompt):
    url = "http://localhost:11434/api/chat"

    payload = {
        "model": "qwen3.5:9b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": True,
        "think": False,
        "keep_alive": "10m"
    }

    answer = ""

    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=(5, 300)
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                chunk = json.loads(line)

                content = chunk.get("message", {}).get("content", "")

                if content:
                    print(content, end="", flush=True)
                    answer += content

        print()
        return answer

    except requests.RequestException as error:
        print(f"\nRequest failed: {error}", file=sys.stderr)
    except json.JSONDecodeError as error:
        print(f"\nInvalid JSON from Ollama: {error}", file=sys.stderr)

    return None


def main():
    print("Waiting for Qwen...\n")

    answer = ask_ollama(
        "Explain the difference between a process and a thread "
        "in three concise bullet points."
    )

    if answer is not None:
        print("\nFinished generating.")


if __name__ == "__main__":
    main()
