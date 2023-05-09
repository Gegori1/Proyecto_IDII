import pandas as pd

def pd_col_to_dummy(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Converts a column to dummy variables and drops the original column
    """
    df_dummy = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, df_dummy], axis=1)
    df = df.drop(col, axis=1)
    return df

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts float to its smallest type
    
    Args:
        df (pd.DataFrame): The dataframe to downcast
        
    Returns:
        pd.DataFrame: A dataframe with float columns downcasted to their smallest type
    
    """
    float_cols = df.select_dtypes(include=["float"]).apply(pd.to_numeric, downcast="float")
    int_cols = df.select_dtypes(include=["int"]).apply(pd.to_numeric, downcast="unsigned")
    str_cols = (
        df.select_dtypes(include=["object"])
        .apply(lambda k: k.astype("category") if k.nunique() < k.shape[0] * 0.3 else k)
        )
    not_converted = df.select_dtypes(exclude=["object", "float", "int"])

    return pd.concat([float_cols, int_cols, str_cols, not_converted], axis=1)[df.columns]


def remove_curp(df:object, path:str)-> object:
  """
  Función para extraer valores importantes de la CURP, remover CURP y guardar DataFrame en path

  Args:
    df: DataFrame con columna CURP

    path: ruta en la que se quiere guardar nuevo DataFrame. Incluye nombre y extension de archivo

  Returns:
    DataFrame limpio

  """

  if "CURP" in df.columns:
    df = (
        df
        # extracción de valores importantes de CURP
        .assign(
            CSexo = lambda k: k.CURP.str[10],
            CEstado = lambda k: k.CURP.str[11:13],
            CFechaNacimiento = lambda k: k.CURP.str[4:10]
        )
        # remover curp
        .drop("CURP", axis=1)
    )
    
    # guradar
    df.to_excel(path, index=False)
  else:
    print("No existe la columna CURP")

  return df

