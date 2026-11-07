import argparse
from pathlib import Path

from learning_analysis import format_learning_report
from problem_similarity import format_similar_problems, similar_problems
from predict_rating import predict_rating
from predict_topics import predict_topics
from recommend_problems import (
    format_recommendations,
    recommend_next_problem,
    recommend_problems,
)


def read_statement(args):
    if args.statement:
        return args.statement

    if args.statement_file:
        return Path(args.statement_file).read_text(encoding="utf-8")

    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Predict Codeforces difficulty and recommend practice problems."
    )
    parser.add_argument(
        "--statement",
        help="Problem statement text to rate.",
    )
    parser.add_argument(
        "--statement-file",
        help="Path to a text file containing the problem statement.",
    )
    parser.add_argument(
        "--target-rating",
        type=int,
        help="Target rating to use when no statement is provided.",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Semicolon-separated tags, for example: 'dp;greedy'.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of recommendations to show.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=300,
        help="Rating window around the predicted or target rating.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show topics, prerequisites, constraints, difficulty, and learning path.",
    )
    parser.add_argument(
        "--current-rating",
        type=int,
        help="Learner's current Codeforces rating for readiness analysis.",
    )
    parser.add_argument(
        "--solved-count",
        type=int,
        default=0,
        help="Number of problems the learner has solved.",
    )
    parser.add_argument(
        "--known-topics",
        default="",
        help="Semicolon-separated topics the learner already knows.",
    )
    parser.add_argument(
        "--similar-to",
        help="Problem name or URL for similar-problem search.",
    )
    parser.add_argument(
        "--next-after",
        help="Solved problem name or URL for next-problem recommendation.",
    )
    parser.add_argument(
        "--predict-topics",
        action="store_true",
        help="Predict Codeforces topics from the statement.",
    )

    args = parser.parse_args()
    statement = read_statement(args)
    tags = [
        tag.strip()
        for tag in args.tags.split(";")
        if tag.strip()
    ]
    known_topics = [
        topic.strip()
        for topic in args.known_topics.split(";")
        if topic.strip()
    ]

    if args.similar_to:
        problems = similar_problems(
            args.similar_to,
            count=args.count,
        )
        print("Similar problems:")
        print(format_similar_problems(problems))
        return

    if args.next_after:
        solved_problem, recommendations = recommend_next_problem(
            args.next_after,
            count=args.count,
        )
        print(
            f"After solving: {solved_problem['name']} "
            f"({int(solved_problem['rating'])})"
        )
        print("\nNext recommended problems:")
        print(format_recommendations(recommendations))
        return

    if statement:
        target_rating = predict_rating(statement, tags=args.tags)
        print(f"Predicted rating: {target_rating}")

        if args.predict_topics or not args.tags:
            predicted_topics = predict_topics(statement)
            print("\nPredicted topics:")
            for item in predicted_topics:
                print(
                    f"- {item['topic']} "
                    f"({round(item['confidence'] * 100, 2)}%)"
                )

            if not args.tags:
                tags = [
                    item["topic"]
                    for item in predicted_topics
                ]
                args.tags = ";".join(tags)
    elif args.target_rating:
        target_rating = args.target_rating
        print(f"Target rating: {target_rating}")
    else:
        parser.error(
            "Provide --statement, --statement-file, or --target-rating."
        )

    if args.analyze:
        print()
        print(
            format_learning_report(
                statement=statement,
                tags=args.tags,
                problem_rating=target_rating,
                user_rating=args.current_rating,
                solved_count=args.solved_count,
                known_topics=known_topics,
            )
        )

    recommendations = recommend_problems(
        target_rating=target_rating,
        weak_tags=tags,
        count=args.count,
        tolerance=args.tolerance,
    )

    print("\nRecommended practice problems:")
    print(format_recommendations(recommendations))


if __name__ == "__main__":
    main()
