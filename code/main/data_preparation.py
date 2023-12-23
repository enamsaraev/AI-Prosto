import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def load_data(file_path: str, sheet: str) -> pd.DataFrame:
    #загружаем нужный файл и лист
    xlsx = pd.ExcelFile(file_path)
    df = xlsx.parse(sheet)

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    #подготавливаем временные фичи
    df['Break flight start'] = pd.to_datetime(df['Break flight start'], format='%H:%M:%S')
    df['Break flight end'] = pd.to_datetime(df['Break flight end'], format='%H:%M:%S')
    df['Programme flight start'] = pd.to_datetime(df['Programme flight start'], format='%H:%M:%S')
    df['Programme flight end'] = pd.to_datetime(df['Programme flight end'], format='%H:%M:%S')

    df['day_of_year'] = df['Date'].dt.dayofyear
    df['день недели'] = df['Date'].dt.day_name()
    df['тип дня'] = df['Date'].dt.dayofweek.apply(lambda x: 'будний' if x < 5 else 'выходной')

    return df


def fix_features(df: pd.DataFrame, column_name: str, name_start: str, name_end: str):
    df[column_name] = (pd.to_datetime(df[name_end]) - pd.to_datetime(df[name_start])).dt.total_seconds() / 60
    # Исправляем длительность рекламы для случаев, когда конец рекламы на следующий день
    for i in range(len(df)):
        if pd.to_datetime(df[name_end][i]) < pd.to_datetime(df[name_start][i]):
            df.loc[i, column_name] = ((pd.to_datetime(df[name_end][i]) + pd.Timedelta(days=1)) - pd.to_datetime(df[name_start][i])).total_seconds() / 60

    return df


def correct_duratiomn(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[(df['Programme flight start'] < df['Programme flight end']) &
           (df['Break flight start'] < df['Programme flight start']), 'начало рекламы'] = -1
    df.loc[(df['Programme flight start'] < df['Programme flight end']) &
           (df['Break flight start'] < df['Programme flight start']), 'начало рекламы'] = -1

    df['длительность рекламы'] = df['длительность рекламы'].round(1)
    df['длительность программы'] = df['длительность программы'].round(1)
    df['начало рекламы'] = df['начало рекламы'].round(1)

    df['между программами'] = np.zeros(len(df))

    df.loc[(df['Programme flight start'] < df['Programme flight end']) &
           (df['Break flight start'] < df['Programme flight start']), 'между программами'] = 1

    df.loc[(df['Programme flight start'] < df['Programme flight end']) &
           (df['Break flight start'] < df['Programme flight start']), 'между программами'] = 1
    
    return df


def start_program_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(7)
    df['начало программы бин'] = pd.cut(
        pd.to_datetime(df['Programme flight start'], format='%H:%M:%S'),
        bins=[
                pd.to_datetime('00:00', format='%H:%M'),
                pd.to_datetime('07:00', format='%H:%M'),
                pd.to_datetime('10:00', format='%H:%M'),
                pd.to_datetime('12:00', format='%H:%M'),
                pd.to_datetime('15:00', format='%H:%M'),
                pd.to_datetime('18:00', format='%H:%M'),
                pd.to_datetime('22:00', format='%H:%M'),
                pd.to_datetime('23:59', format='%H:%M')
            ],
        labels=bins
    )
    df['month'] = df['Date'].dt.strftime('%B')
    
    return df
    

def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    LA = LabelEncoder()

    Label_encoder_cols = [
        'Break content', 
        'Break distribution', 
        'Programme category',
        'Programme genre', 
        'Programme', 
        'день недели', 
        'тип дня', 
        'month'
    ]
    one_hot_cols = ['Programme']
    del_cols = [
        'Date', 
        'Break flight ID', 
        'Break flight start', 
        'Break flight end', 
        'Programme flight start', 
        'Programme flight end'
    ]

    for col in Label_encoder_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    df = df.drop(columns = del_cols)

    df['month'] = LA.fit_transform(df['month'])
    return df


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[
        'day_of_year', 
        'длительность программы', 
        'начало рекламы',
        'длительность рекламы', 
        'день недели', 
        'начало программы бин',
        'Programme', 
        'month', 
        'Programme category', 
        'Break content',
        'Break distribution'
    ]]

    date_t = df[[
        'day_of_year', 
        'день недели', 
        'month'
    ]]

    break_t = df[[
        'начало рекламы', 
        'длительность рекламы', 
        'Break content', 
        'Break distribution'
    ]]

    date_t = PCA(n_components=2).fit_transform(date_t)
    break_t = PCA(n_components=2).fit_transform(break_t)

    date_cluster = KMeans(n_clusters=10).fit(date_t)
    break_cluster = KMeans(n_clusters=4).fit(break_t)

    df['кластер_даты'] = date_cluster.labels_
    df['кластер_рекламы'] = break_cluster.labels_

    return df

    
def prepare(file_path: str, sheet: str) -> pd.DataFrame:
    df = load_data(file_path=file_path, sheet=sheet)
    df = prepare_features(df=df)

    # Исправляем длительность рекламы для случаев, когда конец рекламы на следующий день
    df = fix_features(df=df, column_name='длительность рекламы', name_start='Break flight start', name_end='Break flight end')
    df = fix_features(df=df, column_name='длительность программы', name_start='Programme flight start', name_end='Programme flight end')
    df = fix_features(df=df, column_name='начало рекламы', name_start='Programme flight start', name_end='Break flight end')

    df = start_program_bins(df=df)
    df = make_labels(df=df)
    df = prepare_df(df=df)
    
    df = df.sort_index()
    return df


# res = prepare('code/main/dt.xlsx', 'test data')
# print(res)