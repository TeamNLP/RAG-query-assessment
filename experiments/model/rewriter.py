from typing import Tuple, Optional

from openai import OpenAI


class Rewriter:
    def __init__(self, model_name, system_prompt, rewrite_prompt, input_prompt, max_new_tokens, openai_client):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.rewrite_prompt = rewrite_prompt
        self.input_prompt = input_prompt
        self.max_new_tokens = max_new_tokens
        self.openai_client = openai_client

    def make_rewrite_prompt(
        self,
        query: str
    ) -> Tuple[str, str]:
        rewrite_prompt = self.rewrite_prompt
        input_prompt = self.input_prompt.format(query=query)
        return rewrite_prompt, input_prompt

    def rewrite_query(
        self, 
        query: str
    ) -> Optional[str]:
        rewrite_prompt, input_prompt = self.make_rewrite_prompt(query)
        try:
            response = self.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": rewrite_prompt},
                    {"role": "user", "content": input_prompt}
                ],
                model=self.model_name,
                max_tokens=self.max_new_tokens
            )

            output = response.choices[0].message.content.strip()
            return output

        except Exception as e:
            print(e)
            return None