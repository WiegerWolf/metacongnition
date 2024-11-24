import os
import anthropic

# Initialize the Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
)

def test_llm_call():
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": "Say hello!"}
        ]
    )
    print(message.content)

if __name__ == "__main__":
    test_llm_call()
