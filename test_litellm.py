import dotenv

from litellm import completion

dotenv.load_dotenv()


def main():
    n = 1
    avg = 0
    for i in range(n):
        response = completion(
            model="Gemma-3-12B",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "...",
                },
            ],
            custom_llm_provider="openai",
            # extra_body={
            #     "chat_template_kwargs": '{"reasoning_effort": "high"}',
            # },
            # chat_template_kwargs={"reasoning_effort": "low"},
        )
        print(response)
        print(response.choices[0].finish_reason) # == "length"
        # size = len(response.choices[0].message.reasoning_content)
        # avg += size
        # print(f"[{i=}] {size=} reasoning content")
    avg /= n
    print(f"{avg=}")


if __name__ == "__main__":
    main()
