import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_corr_matrix(corr_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


os.makedirs("outputs", exist_ok=True)

def plot_data(
    df: pd.DataFrame,
    plot_type: str,
    x: str,
    y: str | None = None
) -> None:

    plt.figure(figsize=(8,6))

    if plot_type == 'histogram':
        sns.histplot(df[x], kde=True)  # bins domyślne
        plt.xlabel(x)
        plt.ylabel("Count")
        plt.title(f"Histogram {x}")
        filename = f"histogram_{x}.png"

    elif plot_type == 'boxplot':
        if y is None:
            raise ValueError("Dla boxplot musisz podać argument y (kolumna wartości).")
        sns.boxplot(data=df, x=x, y=y)
        plt.title(f"Boxplot {y} vs {x}")
        filename = f"boxplot_{x}_{y}.png"

    elif plot_type == 'scatter':
        if y is None:
            raise ValueError("Dla scatter musisz podać argument y (kolumna wartości).")
        sns.scatterplot(data=df, x=x, y=y, alpha=0.1) #alpha odpowiada za przezroczystość
        plt.title(f"Scatter {y} vs {x}")
        filename = f"scatter_{x}_{y}.png"

    else:
        raise ValueError("plot_type musi być jednym z: 'histogram', 'boxplot', 'scatter'.")

    plt.tight_layout()

    # zapis do folderu outputs
    filepath = os.path.join("outputs", filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Wykres zapisany: {filepath}")

