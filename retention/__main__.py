import argparse
import json

from .evaluate import evaluate, table


def main():
    p = argparse.ArgumentParser(prog="retention")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--k", type=int, default=3)
    a = p.parse_args()
    report = evaluate(a.n, a.seed, a.k)
    print(table(report))
    print()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
