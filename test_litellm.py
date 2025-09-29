import dotenv

from litellm import completion

dotenv.load_dotenv()


def main():
    n = 10
    avg = 0
    for i in range(n):
        response = completion(
            model="gpt-oss-20B-MXFP4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "What is the 20th term of the Fibonnaci sequence?",
                },
            ],
            custom_llm_provider="openai",
            # extra_body={
            #     "chat_template_kwargs": '{"reasoning_effort": "high"}',
            # },
            chat_template_kwargs={"reasoning_effort": "low"},
        )
        size = len(response.choices[0].message.reasoning_content)
        avg += size
        print(f"[{i=}] {size=} reasoning content")
    avg /= n
    print(f"{avg=}")


if __name__ == "__main__":
    main()
