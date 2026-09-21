import asyncio
from openai import AsyncOpenAI

MODEL = "qwen3.5:4b"
BASE_URL = "http://localhost:11434/v1"

async def test(name, **extra):
    client = AsyncOpenAI(api_key="ollama", base_url=BASE_URL, timeout=60.0)
    messages = [
        {"role": "system", "content": "You must respond with valid JSON only."},
        {"role": "user", "content": 'List 3 statements in JSON: {"statements": ["a","b","c"]}'},
    ]
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
            **extra,
        )
        msg = resp.choices[0].message
        print(f"\n=== {name} ===")
        print("finish_reason:", resp.choices[0].finish_reason)
        print("content:", repr(msg.content)[:300])
        print("reasoning_content:", repr(getattr(msg, "reasoning_content", None))[:300])
        print("usage:", resp.usage)
    except Exception as e:
        print(f"\n=== {name} === ERROR: {type(e).__name__}: {e}")


async def main():
    # 1. 什么都不加，看默认行为
    await test("baseline")

    # 2. reasoning_effort=none（通过 extra_body）
    await test("reasoning_effort=none", extra_body={"reasoning_effort": "none"})

    # 3. chat_template_kwargs 关 thinking
    await test("chat_template_kwargs", extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    # 4. think=False
    await test("think=False", extra_body={"think": False})

    # 5. system prompt 里加 /no_think
    client = AsyncOpenAI(api_key="ollama", base_url=BASE_URL, timeout=60.0)
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "/no_think\nYou must respond with valid JSON only."},
            {"role": "user", "content": 'List 3 statements in JSON: {"statements": ["a","b","c"]}'},
        ],
        max_tokens=512,
    )
    msg = resp.choices[0].message
    print("\n=== system /no_think ===")
    print("finish_reason:", resp.choices[0].finish_reason)
    print("content:", repr(msg.content)[:300])
    print("reasoning_content:", repr(getattr(msg, "reasoning_content", None))[:300])
    print("usage:", resp.usage)


asyncio.run(main())