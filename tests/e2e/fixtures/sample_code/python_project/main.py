def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b


def calculate_product(a, b):
    """Calculate the product of two numbers."""
    return a * b


class Calculator:
    """A simple calculator class for testing code indexing."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        """Add two numbers and store in history."""
        result = calculate_sum(a, b)
        self.history.append({"operation": "add", "a": a, "b": b, "result": result})
        return result

    def multiply(self, a, b):
        """Multiply two numbers and store in history."""
        result = calculate_product(a, b)
        self.history.append({"operation": "multiply", "a": a, "b": b, "result": result})
        return result

    def get_history(self):
        """Return the operation history."""
        return self.history


if __name__ == "__main__":
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"5 * 3 = {calc.multiply(5, 3)}")
    print(f"History: {calc.get_history()}")
