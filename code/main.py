from main.model import make_predict
from main.data_preparation import prepare


def main():
    df = prepare(file_path='code/data/dt.xlsx', sheet='test data')
    predict = make_predict(df=df, model_path='code/model/Light.pgl')

    return predict


if __name__ == '__main__':
    result = main()
    print(result)