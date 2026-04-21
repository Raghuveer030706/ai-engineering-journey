# phase5-production/day18-hardening/cost.py

# Haiku pricing per million tokens (as of 2025)
INPUT_COST_PER_M  = 0.80
OUTPUT_COST_PER_M = 4.00

class CostTracker:
    def __init__(self):
        self.input_tokens  = 0
        self.output_tokens = 0
        self.llm_calls     = 0

    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens  += input_tokens
        self.output_tokens += output_tokens
        self.llm_calls     += 1

    @property
    def total_cost_usd(self) -> float:
        input_cost  = (self.input_tokens  / 1_000_000) * INPUT_COST_PER_M
        output_cost = (self.output_tokens / 1_000_000) * OUTPUT_COST_PER_M
        return round(input_cost + output_cost, 6)

    def summary(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.total_cost_usd,
        }