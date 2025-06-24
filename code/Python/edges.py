import pandas
import numpy
import pickle
import paths
import data_preparation as dp
from os.path import dirname
from sklearn.linear_model import LogisticRegression
import gc


def models_learning(files_in1, files_in2, files_out):
    """
    Обучение классификаторов.

    :param files_in1: Файлы ребер [1й класс, 2й класс].
    :param files_in2: Файлы вершин [1й класс, 2й класс].
    :param files_out: Файлы, куда сохраняются модели.
    :return:
    """
    assert len(files_in1[0]) == len(files_in1[1]) == len(files_in2[0]) == len(files_in2[1])
    data_edge1, data_edge2, data_vertex1, data_vertex2 = [], [], [], []
    for i in range(len(files_in1[0])):
        data_edge1.append(numpy.load(files_in1[0][i]))
        data_edge2.append(numpy.load(files_in1[1][i]))
        data_vertex1.append(pandas.read_pickle(files_in2[0][i])[['mean', 'std']].values)
        data_vertex2.append(pandas.read_pickle(files_in2[1][i])[['mean', 'std']].values)
    data_edge1, data_edge2, data_vertex1, data_vertex2 = (numpy.array(data_edge1), numpy.array(data_edge2),
                                                          numpy.array(data_vertex1), numpy.array(data_vertex2))
    count = 0
    for i in range(379):
        for j in range(i + 1, 379):
            if (i * 379 + j) % 10000 == 0:
                print(i * 379 + j)
            X = pandas.DataFrame({f'v_{i}_mean': numpy.concatenate([data_vertex1[:, i, 0], data_vertex2[:, i, 0]]),
                                  f'v_{j}_mean': numpy.concatenate([data_vertex1[:, j, 0], data_vertex2[:, j, 0]]),
                                  f'v_{i}_std': numpy.concatenate([data_vertex1[:, i, 1], data_vertex2[:, i, 1]]),
                                  f'v_{j}_std': numpy.concatenate([data_vertex1[:, j, 1], data_vertex2[:, j, 1]]),
                                  f'e_{i}_{j}': numpy.concatenate([data_edge1[:, i, j], data_edge2[:, i, j]])})
            Y = [1 for _ in range(data_vertex1.shape[0])] + [2 for _ in range(data_vertex1.shape[0])]

            # model = svm.SVC(kernel="poly", degree=2,  probability=True)
            model = LogisticRegression(random_state=0)
            model.fit(X, Y)
            with open(files_out[count], 'wb') as file:
                pickle.dump(model, file)
            count += 1


def models_calculation(files_in1, files_in2, files_in3, folder_out):
    """
    Вычисление ансамблевых графов.

    :param files_in1: Файлы ребер [1й класс, 2й класс].
    :param files_in2: Файлы вершин [1й класс, 2й класс].
    :param files_in3: Файлы моделей.
    :param folder_out: Папка, куда сохраняются ансамблевые графы.
    :return:
    """
    assert len(files_in1[0]) == len(files_in1[1]) == len(files_in2[0]) == len(files_in2[1])
    data_edge1, data_edge2, data_vertex1, data_vertex2 = [], [], [], []
    for i in range(len(files_in1[0])):
        data_edge1.append(numpy.load(files_in1[0][i]))
        data_edge2.append(numpy.load(files_in1[1][i]))
        data_vertex1.append(pandas.read_pickle(files_in2[0][i])[['mean', 'std']].values)
        data_vertex2.append(pandas.read_pickle(files_in2[1][i])[['mean', 'std']].values)
    data_edge1, data_edge2, data_vertex1, data_vertex2 = (numpy.array(data_edge1), numpy.array(data_edge2),
                                                          numpy.array(data_vertex1), numpy.array(data_vertex2))

    results1 = numpy.zeros(data_edge1.shape)
    results2 = numpy.zeros(data_edge1.shape)
    count = 0
    for i in range(379):
        for j in range(i + 1, 379):
            if (i * 379 + j) % 10000 == 0:
                print(i * 379 + j)
            X = pandas.DataFrame({f'v_{i}_mean': numpy.concatenate([data_vertex1[:, i, 0], data_vertex2[:, i, 0]]),
                                  f'v_{j}_mean': numpy.concatenate([data_vertex1[:, j, 0], data_vertex2[:, j, 0]]),
                                  f'v_{i}_std': numpy.concatenate([data_vertex1[:, i, 1], data_vertex2[:, i, 1]]),
                                  f'v_{j}_std': numpy.concatenate([data_vertex1[:, j, 1], data_vertex2[:, j, 1]]),
                                  f'e_{i}_{j}': numpy.concatenate([data_edge1[:, i, j], data_edge2[:, i, j]])})
            with open(files_in3[count], 'rb') as file:
                model = pickle.load(file)
            Y = model.predict_proba(X)
            results1[:, i, j] = Y[:data_edge1.shape[0], 1] - Y[:data_edge1.shape[0], 0]
            results1[:, j, i] = results1[:, i, j]
            results2[:, i, j] = Y[data_edge1.shape[0]:, 1] - Y[data_edge1.shape[0]:, 0]
            results2[:, j, i] = results2[:, i, j]
            count += 1

    for data1, data2, file1, file2 in zip(results1, results2, files_in1[0], files_in1[1]):
        numpy.save(f'{folder_out}/{file1.split('/')[-1]}', data1)
        numpy.save(f'{folder_out}/{file2.split('/')[-1]}', data2)


def cross_validation(files_in1, files_in2, encoding_type, models_files, folder_out1, folder_out2):
    """
    "Кросс-валидация" – обучение моделей и построение графов для обучения и тестирования ГНС.

    :param files_in1: Файлы ребер.
    :param files_in2: Файлы вершин.
    :param encoding_type: Тип кодирования.
    :param models_files: Файлы моделей.
    :param folder_out1: Папка, куда сохраняются ансамблевые графы для обучения ГНС.
    :param folder_out2: Папка, куда сохраняются ансамблевые графы для тестирования ГНС.
    :return:
    """
    for i in range(4):
        # Строим графы для обучения ГНС.
        for j in range(3):
            print(f'i: {i}, j: {j}')
            # Обучаем ансамбль.
            edge1_tr, edge2_tr = paths.data_preparation(files_in1, encoding_type, paths.folds_ensemble[f'fold{i}'][f'fold{j}']['train'])
            vertex1_tr, vertex2_tr = paths.data_preparation(files_in2, encoding_type, paths.folds_ensemble[f'fold{i}'][f'fold{j}']['train'])
            dp.recreate_folder(dirname(models_files[0]))
            models_learning([edge1_tr, edge2_tr], [vertex1_tr, vertex2_tr], models_files)

            # Строим графы.
            edge1_te, edge2_te = paths.data_preparation(files_in1, encoding_type, paths.folds_ensemble[f'fold{i}'][f'fold{j}']['test'])
            vertex1_te, vertex2_te = paths.data_preparation(files_in2, encoding_type, paths.folds_ensemble[f'fold{i}'][f'fold{j}']['test'])
            dp.recreate_folder(folder_out1[i][j])
            models_calculation([edge1_te, edge2_te], [vertex1_te, vertex2_te], models_files, folder_out1[i][j])
            gc.collect()

        # Строим графы для тестирования ГНС.
        print(f'i: {i}, j: -')
        # Обучаем ансамбль.
        edge1_tr, edge2_tr = paths.data_preparation(files_in1, encoding_type, paths.folds_gnn[f'fold{i}']['train'])
        vertex1_tr, vertex2_tr = paths.data_preparation(files_in2, encoding_type, paths.folds_gnn[f'fold{i}']['train'])
        dp.recreate_folder(dirname(models_files[0]))
        models_learning([edge1_tr, edge2_tr], [vertex1_tr, vertex2_tr], models_files)

        # Строим графы.
        edge1_te, edge2_te = paths.data_preparation(files_in1, encoding_type, paths.folds_gnn[f'fold{i}']['test'])
        vertex1_te, vertex2_te = paths.data_preparation(files_in2, encoding_type, paths.folds_gnn[f'fold{i}']['test'])
        dp.recreate_folder(folder_out2[i])
        models_calculation([edge1_te, edge2_te], [vertex1_te, vertex2_te], models_files, folder_out2[i])
        gc.collect()

