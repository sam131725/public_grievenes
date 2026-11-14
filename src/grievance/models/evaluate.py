import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def per_language_report(
    df: pd.DataFrame, true_col: str, pred_col: str
) -> pd.DataFrame:
    """Return department metrics grouped by the dataset's language column."""
    rows = []
    for language, group in df.groupby("language", sort=True):
        rows.append(
            {
                "language": language,
                "n": len(group),
                "accuracy": accuracy_score(group[true_col], group[pred_col]),
                "macro_f1": f1_score(
                    group[true_col],
                    group[pred_col],
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(rows, columns=["language", "n", "accuracy", "macro_f1"])
