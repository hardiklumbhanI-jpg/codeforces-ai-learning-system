import re
import pandas as pd


def add_structural_features(df):

    # Make a copy so original dataframe is not modified
    df = df.copy()

    # ---------------------------------
    # 1. Character count
    # ---------------------------------
    df["char_count"] = (
        df["statement"]
        .fillna("")
        .apply(len)
    )

    # ---------------------------------
    # 2. Word count
    # ---------------------------------
    df["word_count"] = (
        df["statement"]
        .fillna("")
        .apply(lambda x: len(x.split()))
    )

    # ---------------------------------
    # 3. Sentence count
    # ---------------------------------
    df["sentence_count"] = (
        df["statement"]
        .fillna("")
        .apply(
            lambda x: len(
                re.findall(r"[.!?]+", x)
            )
        )
    )

    # ---------------------------------
    # 4. Number count
    # ---------------------------------
    df["number_count"] = (
        df["statement"]
        .fillna("")
        .apply(
            lambda x: len(
                re.findall(r"\d+", x)
            )
        )
    )

    # ---------------------------------
    # 5. Math symbol count
    # ---------------------------------
    math_symbols = r"[+\-*/=<>%]"

    df["math_symbol_count"] = (
        df["statement"]
        .fillna("")
        .apply(
            lambda x: len(
                re.findall(math_symbols, x)
            )
        )
    )

    # ---------------------------------
    # 6. Average word length
    # ---------------------------------
    def average_word_length(text):

        words = text.split()

        if len(words) == 0:
            return 0

        return sum(len(word) for word in words) / len(words)

    df["avg_word_length"] = (
        df["statement"]
        .fillna("")
        .apply(average_word_length)
    )

    return df