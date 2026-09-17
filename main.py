from sklearn.tree import DecisionTreeClassifier

# [horas_estudadas, nota_simulado]

def import_data(name_file, primary_values, values_predict):

    with open(name_file, 'r') as file:
        for line in file:
            values = [int(x) for x in line.strip().split(",")]

            primary_values.append(values[:-1])
            values_predict.append(values[-1])

def main():

    primary_values = []
    values_predict = []
    
    name_file = 'data.txt'
    import_data(name_file, primary_values, values_predict)

    model = DecisionTreeClassifier()
    model.fit(primary_values, values_predict)

    resultado = model.predict([[2, 30]])
    print(resultado)

if __name__ == '__main__':
    main()