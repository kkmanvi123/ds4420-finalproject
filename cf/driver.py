import argparse

from cf import CollaborativeFiltering as CF

def get_args():
    parser = argparse.ArgumentParser(description="Script to run CF algorithm on a user of interest")
    parser.add_argument("-d", "--data", help="Path to CSV file containing patient data")
    parser.add_argument("-u", "--user", help="User and visitation id formatted as 'user_visit'")
    parser.add_argument("-m", "--metric", help="Metric to calculate similarity scores, one of (cosine, l2)")
    parser.add_argument("-k", help="Value of k nearest patients for predictions (integer)")

    return parser.parse_args()


def main():
    args = get_args()
    model = CF(data=args.data)
    preds = model.run(target=args.user, metric="cosine", k=5, mask_idx=12)
    print(f"Model predictions for patient_visit {args.user}")
    print("-"*100)
    for feat, val in preds.items():
        print(feat, ":\t", val, "\n")

if __name__ == "__main__":
    main()
