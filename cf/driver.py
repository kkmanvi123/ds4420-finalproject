import argparse
import pandas as pd

from cf import CollaborativeFiltering as CF

def get_args():
    parser = argparse.ArgumentParser(description="Script to run CF algorithm on a user of interest")
    parser.add_argument("-d", "--data", help="Path to CSV file containing patient data", required=True)
    parser.add_argument("-u", "--user", help="User and visitation id formatted as 'user_visit'", required=True)
    parser.add_argument("-m", "--metric", help="Metric to calculate similarity scores, one of (cosine, l2)", default="l2")
    parser.add_argument("-k", help="Value of k nearest patients for predictions (integer)", default=11)

    return parser.parse_args()


def main():
    args = get_args()
    df = pd.read_csv(args.data, index_col=0)
    model = CF(data=df)
    preds = model.run(target=args.user, metric=args.metric, k=args.k)
    print(f"Model predictions for patient_visit {args.user}")
    print("-"*100)
    for feat, val in preds.items():
        print(f"{feat}: \t{val}\n")

if __name__ == "__main__":
    main()
