#!/usr/bin/env python
from random import randint
from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl, PositiveFloat
from crewai.flow import Flow, listen, start


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = Field(default=None, ge=0)
    unitPrice: Optional[float] = Field(default=None, ge=0)
    total: float = Field(ge=0)


class Expense(BaseModel):
    merchant: str = Field(description="The name of the merchant or store")
    amount: PositiveFloat = Field(description="The total amount of the expense")
    currency: str = Field(default="USD", description="Currency code, e.g., USD, EUR")
    date: str = Field(description="Date of the expense")
    category: str = Field(description="Expense category")
    categoryId: str = Field(description="Expense category ID")
    imageUrl: HttpUrl = Field(description="Valid image URL is required")
    items: Optional[List[LineItem]] = None
    tax: Optional[float] = None
    tip: Optional[float] = None
    notes: Optional[str] = None


class ExpenseFlowState(BaseModel):
    expense_id: int = 1
    expense_amount: int = 0
    expense_details: Optional[Expense] = None


class ExpenseFlow(Flow[ExpenseFlowState]):

    @start()
    def generate_expense_id(self):
        print("Generating expense id")
        self.state.expense_id = randint(1, 100)

    @listen(generate_expense_id)
    def generate_expense_amount(self):
        print("Generating expense amount")
        self.state.expense_amount = randint(1, 100)

        print("Expense amount generated", self.state.expense_amount)

    @listen(generate_expense_amount)
    def save_expense(self):
        print("Saving expense")
        self.state.expense_details = Expense(
            merchant="Test Merchant",
            amount=self.state.expense_amount,
            currency="USD",
            date="2025-05-30",
            category="Test Category",
            categoryId="test_category_id",
            imageUrl="https://example.com/image.jpg",
        )


def kickoff():
    expense_flow = ExpenseFlow()
    expense_flow.kickoff()


def plot():
    expense_flow = ExpenseFlow()
    expense_flow.plot()


if __name__ == "__main__":
    kickoff()
