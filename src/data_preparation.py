import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target_column: str = "Churn"):
    df = df.copy()

    # Цільова змінна
    y = df[target_column].copy()

    # Ознаки
    X = df.drop(columns=[target_column])

    # Перетворення Churn:
    # No -> 0
    # Yes -> 1
    y = y.replace({
        "No": 0,
        "Yes": 1
    })

    # Перевіряємо результат
    if y.isna().any():
        raise ValueError("У колонці Churn є порожні або невідомі значення.")

    # Перетворюємо текстові колонки у числові
    X = pd.get_dummies(X, drop_first=True)

    # Заповнюємо пропущені значення
    X = X.fillna(0)

    # Перетворюємо цільову змінну у int
    y = y.astype(int)

    return X, y