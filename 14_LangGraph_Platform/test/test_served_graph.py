import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from langgraph_sdk import get_sync_client


def main():
    client = get_sync_client(url="http://localhost:2024")
    for chunk in client.runs.stream(
        None,  # Threadless run
        "simple_agent",  # Assistant id from langgraph.json (assistants)
        input={
            "messages": [
                {
                    "role": "human",
                    "content": "What is the MuonClip optimizer, and what paper did it first appear in?",
                }
            ]
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")


if __name__ == "__main__":
    main()