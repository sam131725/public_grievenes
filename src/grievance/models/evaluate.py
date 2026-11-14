import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from grievance.labels import DEPARTMENTS


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


def department_confusion_matrix(
    df: pd.DataFrame, true_col: str, pred_col: str
) -> pd.DataFrame:
    """Count actual (rows) versus predicted (columns) departments, including absent classes."""
    for column in (true_col, pred_col):
        if not df[column].isin(DEPARTMENTS).all():
            raise ValueError(f"{column} must contain only valid department labels.")
    counts = confusion_matrix(df[true_col], df[pred_col], labels=DEPARTMENTS)
    return pd.DataFrame(
        counts,
        index=pd.Index(DEPARTMENTS, name="actual_department"),
        columns=pd.Index(DEPARTMENTS, name="predicted_department"),
    )


def misclassified_complaints(
    df: pd.DataFrame, true_col: str, pred_col: str
) -> pd.DataFrame:
    """Keep incorrect predictions and their original dataframe indices for review."""
    errors = df.loc[
        df[true_col].ne(df[pred_col]), ["text", "language", true_col, pred_col]
    ].rename(columns={true_col: "actual_department", pred_col: "predicted_department"})
    return errors.rename_axis("source_index").reset_index()
