# W pliku 'plots.py'

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Dodane na wszelki wypadek


def _save_plot(filename: str) -> None:
    try:
        os.makedirs("outputs", exist_ok=True)
        plt.grid(True, linestyle='--', alpha=0.7)

        filepath = os.path.join("outputs", filename)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        print(f"Wykres zapisany: {filepath}")

    except Exception as e:
        print(f"Błąd podczas zapisu wykresu {filename}: {e}")
        plt.close()


def plot_corr_matrix(corr_df: pd.DataFrame, filename: str = "correlation_matrix.png") -> None:
    plt.figure(figsize=(10, 6))

    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Macierz Korelacji", fontsize=14)

    _save_plot(filename)


def plot_data(df: pd.DataFrame, plot_type: str, x: str, y: str | None = None):
    plt.figure(figsize=(10, 6))
    filename = None

    if plot_type == 'histogram':

        sns.histplot(df[x], kde=True, color=plt.get_cmap('Dark2')(0))
        plt.title(f"Rozkład zmiennej: {x}", fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel("Liczba obserwacji (Count)", fontsize=12)
        filename = f"histogram_{x}.png"

    elif plot_type == 'boxplot':
        if y is None:
            raise ValueError("Dla boxplot musisz podać argument y (kolumna wartości).")

        sns.boxplot(data=df, x=x, y=y, palette='Dark2')
        plt.title(f"Boxplot: {y} w zależności od {x}", fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        filename = f"boxplot_{x}_{y}.png"

    elif plot_type == 'scatter':
        if y is None:
            raise ValueError("Dla scatter musisz podać argument y (kolumna wartości).")

        sns.scatterplot(data=df, x=x, y=y, alpha=0.6, color=plt.get_cmap('Dark2')(0))
        plt.title(f"Zależność: {y} vs {x}", fontsize=14)
        plt.xlabel(x, fontsize=12)
        plt.ylabel(y, fontsize=12)
        filename = f"scatter_{x}_{y}.png"

    else:
        raise ValueError("plot_type musi być jednym z: 'histogram', 'boxplot', 'scatter'.")

    if filename:
        _save_plot(filename)