#
# # SPDX-License-Identifier: Apache-2.0
# # Copyright (c) 2026 Data Science Academy Pvt. Ltd. (https://www.datascience.one)
# ##

"""
agents/recommend.py
-------------------
Recommend Agent — sends top products to an LLM and gets a natural language recommendation.

This agent is responsible for:
- Building a prompt from the top ranked products
- Calling the configured LLM provider (Gemini / OpenAI / Anthropic)
- Returning a clean, conversational recommendation response

Students:
- The prompt builder is implemented below.
- You need to complete the LLM call for your chosen provider (see TODOs).
- Only implement the provider your team is using — leave others as stubs.

Testing note:
- Until an LLM provider is implemented, get_recommendation() returns a
  clearly-labeled placeholder response instead of crashing. This lets
  anyone test the Search and Compare agents end-to-end through /chat
  before the LLM integration task is picked up.
"""

import config

PERSONALITY_INSTRUCTIONS = {
    "Tech Geek": (
        "Use a technically detailed style. Prioritize processor, RAM, storage, "
        "display, camera, battery capacity, performance, and 5G details that "
        "are present in the supplied product data. Compare trade-offs."
    ),
    "Bargain Hunter": (
        "Prioritize the supplied prices, budget fit, value, and price comparisons. "
        "Mention a discount only when both supplied price and MRP establish it; "
        "never invent a deal, saving, or price."
    ),
    "Everyday User": (
        "Use friendly, accessible language with little jargon. Focus on practical "
        "daily use, battery life, ease of use, reliability, camera usability, and "
        "overall suitability only where supported by the supplied product details."
    ),
}


def build_prompt(
    user_query: str,
    top_products: list[dict],
    personality: str = "Everyday User",
) -> str:
    """
    Builds the prompt to send to the LLM.

    Args:
        user_query   : Original query from the user
        top_products : Top ranked products from the Compare Agent
        personality : Response style selected by the user

    Returns:
        str: Formatted prompt string
    """
    product_lines = []
    for i, p in enumerate(top_products, start=1):
        product_lines.append(
            f"{i}. {p['product_name']} by {p['brand']}\n"
            f"   Store: {p['store']} | Price: Rs {p['price']} (MRP: Rs {p['mrp']})\n"
            f"   Rating: {p['rating']}/5 ({p['review_count']} reviews)\n"
            f"   Specs: {p['key_specs']}\n"
            f"   Pros: {p['pros']}\n"
            f"   Cons: {p['cons']}\n"
            f"   Score: {p.get('score', 'N/A')}/100"
        )

    products_text = "\n\n".join(product_lines)

    instructions = PERSONALITY_INSTRUCTIONS[personality]
    prompt = f"""You are a helpful shopping assistant. A user is looking for: "{user_query}"
Selected personality: {personality}
Personality instructions: {instructions}

Based on the available products, here are the top recommendations:

{products_text}

Please give a friendly, concise recommendation in 3–5 sentences.
- Mention which product offers the best value and why.
- Point out any trade-offs the user should know about.
- Suggest which store to buy from based on the price.
- Do not make up any information not provided above.
"""
    return prompt


def _build_fallback_response(
    provider: str,
    top_products: list[dict],
    personality: str = "Everyday User",
) -> str:
    """
    Returns a clearly-labeled placeholder response when the LLM provider
    has not been implemented yet (_call_gemini etc. raises NotImplementedError).

    This allows the Search and Compare agents to be tested end-to-end
    through the /chat endpoint without requiring the LLM integration
    task to be finished first.

    Args:
        provider     : The configured LLM_PROVIDER name (gemini/openai/anthropic)
        top_products : Top ranked products from the Compare Agent
        personality  : Response style selected by the user

    Returns:
        str: A placeholder message listing the top products with a note
             that the LLM call still needs to be implemented.
    """
    product_summary = "\n".join(
        f"  {i + 1}. {p['product_name']} ({p['store']}) "
        f"— Rs {p['price']} — score {p.get('score', 'N/A')}"
        for i, p in enumerate(top_products)
    )
    top_product = top_products[0]
    if personality == "Tech Geek":
        focus = (
            f"Technical focus: {top_product['product_name']} — listed specs: "
            f"{top_product['key_specs']}"
        )
    elif personality == "Bargain Hunter":
        prices = "; ".join(
            f"{product['product_name']}: Rs {product['price']}"
            + (
                f" (listed MRP Rs {product['mrp']})"
                if product.get("mrp") and product["mrp"] > product["price"]
                else ""
            )
            for product in top_products
        )
        focus = f"Price and value focus: listed prices are {prices}."
    else:
        focus = (
            f"Everyday-use snapshot: {top_product['product_name']} is the top-ranked "
            f"match in this result set at Rs {top_product['price']}, rated "
            f"{top_product['rating']}/5 from {top_product['review_count']} reviews."
        )
    return (
        f"[LLM not yet connected — implement _call_{provider}() in agents/recommend.py]\n\n"
        f"Pipeline is working correctly. Top {len(top_products)} results found:\n"
        f"{product_summary}\n\n{focus}"
    )


def get_recommendation(
    user_query: str,
    top_products: list[dict],
    personality: str = "Everyday User",
) -> str:
    """
    Sends the top products to the configured LLM and returns a recommendation.

    If the configured LLM provider has not been implemented yet, returns a
    clearly-labeled placeholder instead of crashing — this keeps the
    Search and Compare agents testable through /chat at all times.

    Args:
        user_query   : Original query from the user
        top_products : Top ranked products from the Compare Agent
        personality  : Response style selected by the user

    Returns:
        str: Natural language recommendation from the LLM, or a
             placeholder message if the LLM integration is not yet built.

    Raises:
        ValueError: If LLM_PROVIDER is not recognised
    """
    if not top_products:
        return "Sorry, I could not find any products matching your request. Try a different search."

    prompt = build_prompt(user_query, top_products, personality)
    provider = config.LLM_PROVIDER.lower()

    if provider not in ("gemini", "openai", "anthropic"):
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. Set it to gemini, openai, or anthropic in .env"
        )

    try:
        if provider == "gemini":
            return _call_gemini(prompt)
        elif provider == "openai":
            return _call_openai(prompt)
        elif provider == "anthropic":
            return _call_anthropic(prompt)
    except NotImplementedError:
        return _build_fallback_response(provider, top_products, personality)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Provider Implementations
# ─────────────────────────────────────────────────────────────────────────────


def _call_gemini(prompt: str) -> str:
    """
    Calls the Google Gemini API and returns the generated response.

    Args:
        prompt: The recommendation prompt.

    Returns:
        The generated recommendation text.

    Raises:
        ValueError: If the API key is missing or the response is empty.
        RuntimeError: If the Gemini API request fails.
    """
    from google import genai

    api_key = config.GEMINI_API_KEY

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Please configure it in your .env file."
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        response_text = response.text

        if not response_text or not response_text.strip():
            raise ValueError("Gemini returned an empty response.")

        return response_text.strip()

    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Gemini API request failed: {type(exc).__name__}"
        ) from exc




def _call_openai(prompt: str) -> str:
    """
    Calls the OpenAI API and returns the response text.

    TODO:
        1. Install the SDK:       pip install openai
        2. Import and call:
               from openai import OpenAI
               client = OpenAI(api_key=config.OPENAI_API_KEY)
               response = client.chat.completions.create(
                   model="gpt-4o-mini",
                   messages=[{"role": "user", "content": prompt}]
               )
               return response.choices[0].message.content
    """
    raise NotImplementedError("OpenAI integration not yet implemented. See TODO in _call_openai()")


def _call_anthropic(prompt: str) -> str:
    """
    Calls the Anthropic Claude API and returns the response text.

    TODO:
        1. Install the SDK:       pip install anthropic
        2. Import and call:
               import anthropic
               client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
               message = client.messages.create(
                   model="claude-haiku-4-5-20251001",
                   max_tokens=512,
                   messages=[{"role": "user", "content": prompt}]
               )
               return message.content[0].text
    """
    raise NotImplementedError("Anthropic integration not yet implemented. See TODO in _call_anthropic()")
