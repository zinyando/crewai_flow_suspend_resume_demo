#!/usr/bin/env python
from random import randint
from typing import List, Optional
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
import uuid

import litellm
from pydantic import BaseModel, Field, HttpUrl, PositiveFloat
from dateutil import parser as date_parser

if not os.getenv("OPENAI_API_KEY"):
    print(
        "Warning: OPENAI_API_KEY environment variable not set. LiteLLM calls to OpenAI will fail."
    )
from crewai.flow import Flow, listen, start, router


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = Field(default=None, ge=0)
    unitPrice: Optional[float] = Field(default=None, ge=0)
    total: float = Field(ge=0)


class Expense(BaseModel):
    merchant: str = Field(description="The name of the merchant or store")
    amount: PositiveFloat = Field(description="The total amount of the expense")
    currency: str = Field(default="USD", description="Currency code, e.g., USD, EUR")
    date: str = Field(
        description="Date of the expense"
    )  # Should be ISO format after extraction
    category: str = Field(description="Expense category")
    categoryId: str = Field(description="Expense category ID")
    imageUrl: HttpUrl = Field(description="Valid image URL is required")
    items: Optional[List[LineItem]] = None
    tax: Optional[float] = None
    tip: Optional[float] = None
    notes: Optional[str] = None
    # Fields for saved expense
    id: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None


class ExpenseFlowState(BaseModel):
    id: int = 1
    expense_details: Optional[Expense] = None
    input_image_url: Optional[HttpUrl] = None


class ExpenseFlow(Flow[ExpenseFlowState]):

    @start()
    async def load_data(self):
        print("Expense load data")
        # load expense details from postgres

    @router()
    def select_step(self):
        return self.state.id

    @listen(select_step)
    async def extract_expense(self):
        print("Extracting expense data from image...")
        if not self.state.input_image_url:
            print("Error: input_image_url is not set in the state.")
            return

        image_url_str = str(self.state.input_image_url)

        system_prompt = (
            "You are an image data extractor assistant. Extract all expense information from the provided receipt image "
            "in JSON format. The JSON output MUST conform to the following Pydantic schema structure:\n"
            "Expense Schema:\n"
            "  merchant: str (The name of the merchant or store)\n"
            "  amount: float (The total amount of the expense, must be positive)\n"
            "  currency: str (Currency code, e.g., USD, EUR; default to 'USD' if not found)\n"
            "  date: str (Date of the expense, extract as seen, will be normalized later)\n"
            "  category: str (Expense category, e.g., Food, Travel, Office Supplies)\n"
            "  categoryId: str (Expense category ID)\n"
            "  imageUrl: str (This will be the original input image URL, you don't need to find it in the image itself)\n"
            "  items: Optional list of LineItem objects\n"
            "  tax: Optional float (default to 0.0 if not found)\n"
            "  tip: Optional float (default to 0.0 if not found)\n"
            "  notes: Optional str (default to empty string if not found)\n"
            "LineItem Schema (for each item in 'items'):\n"
            "  description: str\n"
            "  quantity: Optional float (default to 1.0 if not applicable or not found, must be >= 0)\n"
            "  unitPrice: Optional float (default to 1.0 if not applicable or not found, must be >= 0)\n"
            "  total: float (must be >= 0)\n\n"
            "Important instructions:\n"
            "- For the 'date', extract it exactly as it appears on the receipt.\n"
            "- For the 'amount' and other numerical fields, extract just the number (no currency symbols).\n"
            "- Ensure all number fields are actual numbers, not strings.\n"
            "- If 'quantity' or 'unitPrice' for an item is not available or not applicable, set them to 1.0.\n"
            "- If 'tax' or 'tip' is not found, set them to 0.0.\n"
            "- If 'notes' are not found, set it to an empty string.\n"
            "- If 'items' are not clearly discernible, you can omit the 'items' field or provide an empty list."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract expense data from this image."},
                    {"type": "image_url", "image_url": {"url": image_url_str}},
                ],
            },
        ]

        try:
            print(f"Calling LLM for image URL: {image_url_str}")
            response = await litellm.acompletion(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            raw_json_output = response.choices[0].message.content
            print(f"Raw LLM JSON output: {raw_json_output}")
            extracted_data = json.loads(raw_json_output)

            extracted_data["imageUrl"] = image_url_str

            if "date" in extracted_data and extracted_data["date"]:
                original_date_str = extracted_data["date"]
                if "T" not in original_date_str:
                    try:
                        parsed_date = date_parser.parse(original_date_str)
                        extracted_data["date"] = parsed_date.isoformat()
                        print(
                            f"Normalized date from '{original_date_str}' to '{extracted_data['date']}'"
                        )
                    except (ValueError, TypeError, date_parser.ParserError) as e:
                        print(
                            f"Warning: Could not convert date '{original_date_str}' to ISO format: {e}. Using original."
                        )

            self.state.expense_details = Expense(**extracted_data)
            print(
                f"Successfully extracted and validated expense data: {self.state.expense_details.model_dump_json(indent=2)}"
            )

        except litellm.exceptions.APIConnectionError as e:
            print(f"LiteLLM API Connection Error: {e}")
        except litellm.exceptions.RateLimitError as e:
            print(f"LiteLLM Rate Limit Error: {e}")
        except litellm.exceptions.APIError as e:
            print(f"LiteLLM API Error: {e}")
        except json.JSONDecodeError as e:
            raw_output_for_error = locals().get("raw_json_output", "N/A")
            print(
                f"Error decoding JSON from LLM response: {e}. Response was: {raw_output_for_error}"
            )
        except Exception as e:
            print(f"An unexpected error occurred during expense extraction: {e}")

    @listen(extract_expense)
    async def categorize_expense(self):
        print("Categorizing expense...")
        if not self.state.expense_details:
            print("Error: Expense details not found for categorization.")
            return

        available_categories = [
            {"id": "food_drink", "name": "Food & Drink"},
            {"id": "transport", "name": "Transportation"},
            {"id": "shopping", "name": "Shopping"},
            {"id": "services", "name": "Services"},
            {"id": "utilities", "name": "Utilities"},
            {"id": "travel", "name": "Travel"},
            {"id": "health_wellness", "name": "Health & Wellness"},
            {"id": "entertainment", "name": "Entertainment"},
            {"id": "education", "name": "Education"},
            {"id": "other", "name": "Other"},
        ]
        category_names = [c["name"] for c in available_categories]

        merchant_name = self.state.expense_details.merchant
        item_descriptions = "N/A"
        if self.state.expense_details.items:
            item_descriptions = ", ".join(
                [item.description for item in self.state.expense_details.items]
            )

        system_prompt = f"You are an expense categorization assistant. Categorize the following expense into one of these categories: {', '.join(category_names)}. Return only the category name."
        user_content = f"Merchant: {merchant_name}, Items: {item_descriptions}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = await litellm.acompletion(
                model="gpt-4o", messages=messages, temperature=0.1
            )
            suggested_category_name = response.choices[0].message.content.strip()
            print(f"LLM suggested category: {suggested_category_name}")

            best_match = next(
                (
                    c
                    for c in available_categories
                    if c["name"].lower() == suggested_category_name.lower()
                ),
                None,
            )
            if not best_match:
                best_match = next(c for c in available_categories if c["id"] == "other")

            self.state.expense_details.category = best_match["name"]
            self.state.expense_details.categoryId = best_match["id"]
            print(
                f"Expense categorized as: {self.state.expense_details.category} (ID: {self.state.expense_details.categoryId})"
            )

        except Exception as e:
            print(f"An error occurred during expense categorization: {e}")
            other_category = next(c for c in available_categories if c["id"] == "other")
            self.state.expense_details.category = other_category["name"]
            self.state.expense_details.categoryId = other_category["id"]
            print(f"Defaulted to category: {self.state.expense_details.category}")

    @listen(categorize_expense)
    def review_expense(self):
        print("\n--- Expense Review --- বর্ণের")
        if self.state.expense_details:
            print("Please review the extracted and categorized expense details:")
            print(self.state.expense_details.model_dump_json(indent=2))
            print("This is a point where the flow could be suspended.")
            print(
                "If changes are needed, the state can be updated before resuming to the 'save_expense' step."
            )
        else:
            print("Error: No expense details to review.")
        print("--- End of Review ---\n")

    @listen(review_expense)
    def save_expense(self):
        print("Saving expense...")
        if not self.state.expense_details:
            print("Error: No expense details to save.")
            return

        now = datetime.now(timezone.utc)
        self.state.expense_details.id = str(uuid.uuid4())
        self.state.expense_details.createdAt = now
        self.state.expense_details.updatedAt = now

        try:
            with open("expense_details.json", "w") as f:
                f.write(self.state.expense_details.model_dump_json(indent=2))
            print(
                f"Expense saved successfully with ID: {self.state.expense_details.id}"
            )
            print("Full saved details:")
            print(self.state.expense_details.model_dump_json(indent=2))
        except IOError as e:
            print(f"Error saving expense to file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during saving: {e}")


def kickoff():
    expense_flow = ExpenseFlow()
    asyncio.run(expense_flow.kickoff())


def plot():
    expense_flow = ExpenseFlow()
    expense_flow.plot()


if __name__ == "__main__":
    kickoff()
