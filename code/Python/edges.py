import pandas
import numpy
import pickle
from sklearn import svm
import multiprocessing


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
            if (i * 379 + j) % 1000 == 0:
                print(i * 379 + j)
            X = pandas.DataFrame({f'v_{i}_mean': numpy.concatenate([data_vertex1[:, i, 0], data_vertex2[:, i, 0]]),
                                  f'v_{j}_mean': numpy.concatenate([data_vertex1[:, j, 0], data_vertex2[:, j, 0]]),
                                  f'v_{i}_std': numpy.concatenate([data_vertex1[:, i, 1], data_vertex2[:, i, 1]]),
                                  f'v_{j}_std': numpy.concatenate([data_vertex1[:, j, 1], data_vertex2[:, j, 1]]),
                                  f'e_{i}_{j}': numpy.concatenate([data_edge1[:, i, j], data_edge1[:, i, j]])})
            Y = [1 for _ in range(data_vertex1.shape[0])] + [2 for _ in range(data_vertex1.shape[0])]

            svc = svm.SVC(kernel="poly", probability=True)
            svc.fit(X, Y)
            pickle.dump(svc, open(files_out[count], 'wb'))
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
            if (i * 379 + j) % 1000 == 0:
                print(i * 379 + j)
            X = pandas.DataFrame({f'v_{i}_mean': numpy.concatenate([data_vertex1[:, i, 0], data_vertex2[:, i, 0]]),
                                  f'v_{j}_mean': numpy.concatenate([data_vertex1[:, j, 0], data_vertex2[:, j, 0]]),
                                  f'v_{i}_std': numpy.concatenate([data_vertex1[:, i, 1], data_vertex2[:, i, 1]]),
                                  f'v_{j}_std': numpy.concatenate([data_vertex1[:, j, 1], data_vertex2[:, j, 1]]),
                                  f'e_{i}_{j}': numpy.concatenate([data_edge1[:, i, j], data_edge1[:, i, j]])})
            svm_ = pickle.load(open(files_in3[count], 'rb'))
            Y = svm_.predict_proba(X)
            results1[:, i, j] = Y[:data_edge1.shape[0], 1] - Y[:data_edge1.shape[0], 0]
            results2[:, i, j] = Y[data_edge1.shape[0]:, 1] - Y[data_edge1.shape[0]:, 0]
            count += 1

    for data1, data2, file1, file2 in zip(results1, results2, files_in1[0], files_in1[1]):
        numpy.save(f'{folder_out}/{file1.split('/')[-1]}', data1)
        numpy.save(f'{folder_out}/{file2.split('/')[-1]}', data2)

ппереоформить каталог ансаблевых графов в словарь и написать цикл вычисления всего подряд
# --------------------------------------


def ensemble_graphs_(files_in1, files_in2, files_in3, files_out):
    def ensemble_graphs_(files_in1, files_in2, files_in3, files_out, proc):
        data_vertex = pandas.read_csv(files_in2[0], index_col=0)
        data_edge = pandas.read_csv(files_in3[0], index_col=[0, 1])
        for i in range(1, len(files_in2)):
            data_vertex = pandas.concat([data_vertex, pandas.read_csv(files_in2[i], index_col=0)], axis=1)
            data_edge = pandas.concat([data_edge, pandas.read_csv(files_in3[i], index_col=[0, 1])], axis=1)
        vertexes = [[], []]

        count = 0
        for i in range(379):
            for j in range(i + 1, 379):
                vertexes[0].append(i + 1), vertexes[1].append(j + 1)

                svm = pickle.load(open(files_in1[count], 'rb'))
                X = pandas.DataFrame({f"v{i + 1}": data_vertex.iloc[i, :].tolist(),
                                      f"v{j + 1}": data_vertex.iloc[j, :].tolist(),
                                      f"e{i + 1}_{j + 1}": data_edge.loc[(i, j), :].tolist()})
                Y = svm.predict_proba(X)

                if i + j == 1:
                    data_edge_new = Y[:, 1] - Y[:, 0]
                else:
                    data_edge_new = numpy.vstack((data_edge_new, Y[:, 1] - Y[:, 0]))

        for count, file_out in enumerate(files_out):
            graph = pandas.DataFrame({"vertex1": vertexes[0], "vertex2": vertexes[1], "edge_weight": data_edge_new[:, count]})
            graph.to_csv(file_out, index=False)

    files_in1_LR = files_in1[:71631]
    files_in1_RL = files_in1[71631:]
    files_in2_LR_1 = files_in2[::4]
    files_in2_LR_2 = files_in2[1::4]
    files_in2_RL_1 = files_in2[2::4]
    files_in2_RL_2 = files_in2[3::4]
    files_in3_LR_1 = files_in3[::4]
    files_in3_LR_2 = files_in3[1::4]
    files_in3_RL_1 = files_in3[2::4]
    files_in3_RL_2 = files_in3[3::4]
    files_out_LR_1 = files_out[::4]
    files_out_LR_2 = files_out[1::4]
    files_out_RL_1 = files_out[2::4]
    files_out_RL_2 = files_out[3::4]

    processes = []
    processes.append(multiprocessing.Process(target=ensemble_graphs_,
                                             args=(files_in1_LR, files_in2_LR_1, files_in3_LR_1, files_out_LR_1, "LR 1")))
    processes.append(multiprocessing.Process(target=ensemble_graphs_,
                                             args=(files_in1_LR, files_in2_LR_2, files_in3_LR_2, files_out_LR_2, "LR 2")))
    processes.append(multiprocessing.Process(target=ensemble_graphs_,
                                             args=(files_in1_RL, files_in2_RL_1, files_in3_RL_1, files_out_RL_1, "RL 1")))
    processes.append(multiprocessing.Process(target=ensemble_graphs_,
                                             args=(files_in1_RL, files_in2_RL_2, files_in3_RL_2, files_out_RL_2, "RL 2")))
    for process in processes:
        process.start()
    for process in processes:
        process.join()