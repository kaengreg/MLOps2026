from __future__ import annotations
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth

from config import METRICS_DIR, REPORTS_DIR, TARGET_COL
from src.data_prep import list_batch_files, load_batch
from src.features import prepare_dataset


ASSOCIATION_RULES_FILE = METRICS_DIR / "association_rules.csv"
ASSOCIATION_RULES_REPORT_FILE = REPORTS_DIR / "association_rules.md"


def load_training_batches(batch_files=None, max_batches=None):
    if batch_files is None:
        batch_files = list_batch_files()

    if max_batches is not None:
        if max_batches < 1:
            raise ValueError("max_batches must be at least 1.")
        batch_files = batch_files[:max_batches]

    if not batch_files:
        raise ValueError("No batch files found for association rules generation.")

    dfs = []
    for batch_path in batch_files:
        df = load_batch(batch_path)
        df["source_batch"] = batch_path.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def build_binary_conditions(df):
    local_df = prepare_dataset(df)

    q75_trip_distance = local_df["trip_distance"].quantile(0.75) if "trip_distance" in local_df.columns else 0
    q75_trip_duration = local_df["trip_duration_min"].quantile(0.75) if "trip_duration_min" in local_df.columns else 0
    q75_total_amount = local_df[TARGET_COL].quantile(0.75) if TARGET_COL in local_df.columns else 0

    binary_df = pd.DataFrame(
        {
            "pickup_night": (local_df["pickup_hour"] < 6) | (local_df["pickup_hour"] >= 22),
            "pickup_weekend": local_df["pickup_weekday"] >= 5,
            "trip_distance_high": local_df["trip_distance"] > q75_trip_distance,
            "trip_duration_high": local_df["trip_duration_min"] > q75_trip_duration,
            "passenger_many": local_df["passenger_count"] >= 4,
            "total_amount_high": local_df[TARGET_COL] > q75_total_amount,
        }
    )

    return binary_df.astype(bool)


def generate_association_rules(min_support: float = 0.05, min_confidence: float = 0.6, max_batches: int = None, top_k: int = 5):
    raw_df = load_training_batches(max_batches=max_batches)
    binary_df = build_binary_conditions(raw_df)

    frequent_itemsets = fpgrowth(binary_df, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        raise ValueError("No frequent itemsets were found. Try lowering min_support or using more batch files.")

    rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules_df.empty:
        raise ValueError("No association rules were found. Try lowering min_confidence or min_support.")

    rules_df = rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].copy()

    def format_itemset(itemset) -> str:
        return " & ".join(sorted(map(str, itemset)))

    rules_df["A_text"] = rules_df["antecedents"].apply(format_itemset)
    rules_df["B_text"] = rules_df["consequents"].apply(format_itemset)
    rules_df["rule"] = rules_df["A_text"] + " -> " + rules_df["B_text"]

    rules_df = rules_df.sort_values(by=["lift", "confidence", "support"], ascending=[False, False, False]).reset_index(drop=True)
    rules_df = rules_df.drop_duplicates(subset=["rule"]).reset_index(drop=True)
    top_rules_df = rules_df.head(top_k).copy()

    save_rules_outputs(
        all_rules_df=rules_df,
        top_rules_df=top_rules_df,
        min_support=min_support,
        min_confidence=min_confidence,
        batch_count=len(list_batch_files()[:max_batches]) if max_batches is not None else len(list_batch_files()),
        row_count=len(binary_df),
    )

    return {
        "all_rules_file": ASSOCIATION_RULES_FILE,
        "report_file": ASSOCIATION_RULES_REPORT_FILE,
        "rules_found": int(len(rules_df)),
        "top_rules_selected": int(len(top_rules_df)),
        "top_rules": top_rules_df[["rule", "support", "confidence", "lift"]].to_dict(orient="records"),
    }


def save_rules_outputs(all_rules_df, top_rules_df, min_support, min_confidence, batch_count, row_count):
    ASSOCIATION_RULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ASSOCIATION_RULES_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    export_df = all_rules_df[["rule", "support", "confidence", "lift", "A_text", "B_text"]].copy()
    export_df.to_csv(ASSOCIATION_RULES_FILE, index=False)

    lines = [
        "# Association Rules Report",
        "",
        "## Configuration",
        "",
        f"- min_support: {min_support}",
        f"- min_confidence: {min_confidence}",
        f"- batch_count: {batch_count}",
        f"- row_count_after_preparation: {row_count}",
        "",
        "## Purpose",
        "",
        "Правила используются как вспомогательные паттерны для проверки правдоподобия и корректности данных.",
        "",
        "## Top-5 rules",
        "",
    ]

    for idx, row in top_rules_df.reset_index(drop=True).iterrows():
        lines.extend(
            [
                f"### Rule {idx + 1}",
                "",
                f"- rule: `{row['rule']}`",
                f"- support: {row['support']:.4f}",
                f"- confidence: {row['confidence']:.4f}",
                f"- lift: {row['lift']:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Full rules table",
            "",
            export_df.head(20).to_markdown(index=False),
            "",
        ]
    )

    ASSOCIATION_RULES_REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")