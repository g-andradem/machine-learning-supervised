from sklearn.tree import DecisionTreeClassifier
import numpy as np

def import_data(name_file):

    primary_values = []
    values_predict = []

    with open(name_file, 'r') as file:
        for line in file:
            values = [int(x) for x in line.strip().split(",")]

            primary_values.append(values[:-1])
            values_predict.append(values[-1])

    print(primary_values)
    print(values_predict)

    return np.array(primary_values), np.array(values_predict)

def main_example():
    
    name_file = 'data.txt'
    primary_values, values_predict = import_data(name_file)

    model = DecisionTreeClassifier()
    model.fit(primary_values, values_predict)

    resultado = model.predict([[8, 16, 24]])
    print(resultado)

if __name__ == '__main__':
    main_example()