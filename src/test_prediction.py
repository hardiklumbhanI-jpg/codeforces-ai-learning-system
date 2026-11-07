from predict_rating import predict_rating

problem = """
You are given an array of n integers.
Find the maximum possible sum after performing operations.
"""

print(
    "Predicted Rating:",
    predict_rating(problem, tags="greedy;dp")
)
