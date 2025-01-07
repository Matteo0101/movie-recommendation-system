import pandas as pd

def load_data(file_path, sep='\t', columns=None, encoding='utf-8'):
    """
    Carica il dataset da un file e lo restituisce come DataFrame pandas.

    Args:
        file_path (str): Percorso del file.
        sep (str): Separatore dei campi nel file.
        columns (list): Nomi delle colonne del DataFrame.
        encoding (str): Codifica del file (default: 'utf-8').

    Returns:
        pd.DataFrame: Il dataset caricato.
    """
    df = pd.read_csv(file_path, sep=sep, names=columns, encoding=encoding)
    return df


def create_user_item_matrix(df, user_col='user_id', item_col='item_id', rating_col='rating'):
    """
    Crea una matrice user-item (punteggi dati dagli utenti agli item).

    Args:
        df (pd.DataFrame): DataFrame con i dati (utenti, item, rating).
        user_col (str): Nome della colonna degli utenti.
        item_col (str): Nome della colonna degli item.
        rating_col (str): Nome della colonna dei rating.

    Returns:
        pd.DataFrame: Matrice user-item.
    """
    matrix = df.pivot(index=user_col, columns=item_col, values=rating_col)
    matrix.fillna(0, inplace=True)  # Riempi NaN con 0
    return matrix