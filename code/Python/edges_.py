import numpy
import pandas
import multiprocessing

from matplotlib import pyplot as plt
from pickle import load as pickle_load
from scipy.stats import rankdata, linregress
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.tsa.stattools import grangercausalitytests
# from pyinform.transferentropy import transfer_entropy#   as transfer_entropy_
# import pyinform.transferentropy


def cross_correlation_(data, lag):
    """
    Вычисление pearson cross correlation.

    :param data: Данные временных рядов.
    :param lag: Сдвиг.
    :return: Корреляционная матрица
    """
    result = []
    for count, block in enumerate(data):
        # print(count)
        if lag == 0:
            result_block = numpy.corrcoef(block)
        else:
            result_block = numpy.zeros((block.shape[0], block.shape[0]))
            for i in range(block.shape[0]):
                for j in range(i + 1, block.shape[0]):
                    array = [numpy.corrcoef(block[i, :], block[j, :])[0, 1]]  # нет сдвига
                    for k in range(1, lag + 1):
                        array.append(numpy.corrcoef(block[i, :-k], block[j, k:])[0, 1])  # i после j
                        array.append(numpy.corrcoef(block[i, k:], block[j, :-k])[0, 1])  # i до j
                    result_block[i, j] = array[numpy.abs(array).argmax()]  # Максимальное число по абсолютному значению
                    result_block[j, i] = result_block[i, j]  # Матрица симметричная

        mean = result_block[numpy.triu_indices(block.shape[0], k=1)].mean()
        std = result_block[numpy.triu_indices(block.shape[0], k=1)].std()
        result_block = (result_block - mean) / std
        result_block[numpy.diag_indices(result_block.shape[0])] = 0
        result.append(result_block)

    return numpy.mean(result, axis=0)


def pearson_cross_correlation(file_in, file_out, lag=0):
    """
    Вычисление pearson cross correlation.

    :param file_in: Файл временных рядов регионов.
    :param file_out: Файл ребер.
    :param lag: Сдвиг.
    :return:
    """
    print(file_in, file_out)
    with open(file_in, 'rb') as file:
        data = pickle_load(file)
        data = [block.to_numpy() for block in data]
        result = cross_correlation_(data, lag)
        numpy.save(file_out, result)

def spearman_cross_correlation(file_in, file_out, lag=0):
    """
    Вычисление spearman cross correlation.

    :param file_in: Файл временных рядов регионов.
    :param file_out: Файл ребер.
    :param lag: Сдвиг.
    :return:
    """
    with open(file_in, 'rb') as file:
        data = pickle_load(file)
        data = [rankdata(block, method='dense', axis=1) for block in data]
        result = cross_correlation_(data, lag)
        numpy.savetxt(file_out, result)


def mutual_information(file_in, file_out, lag=0, n_bins=4, strategy='quantile'):
    """
    Вычисление mutual information.

    :param file_in: Файл временных рядов регионов.
    :param file_out: Файл ребер.
    :param lag: Сдвиг.
    :param n_bins: Число кластеров дискретизации.
    :param strategy: Стратегия дискретизации.
    :return:
    """
    print(file_in, file_out)
    with open(file_in, 'rb') as file:
        data = pickle_load(file)
        data = [block.to_numpy() for block in data]

    result = []
    for count, block in enumerate(data):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        for i in range(block.shape[0]):
            block[i, :] = discretizer.fit_transform(block[i, :].reshape(-1, 1)).flatten()

        result_block = numpy.zeros((block.shape[0], block.shape[0]))
        for i in range(block.shape[0]):
            for j in range(i + 1, block.shape[0]):
                # if (379 * i + j) % 100 == 0:
                #     print(379 * i + j)
                array = [mutual_info_score(block[i, :], block[j, :])]  # нет сдвига
                for k in range(1, lag + 1):
                    array.append(mutual_info_score(block[i, :-k], block[j, k:]))  # i после j
                    array.append(mutual_info_score(block[i, k:], block[j, :-k]))  # i до j
                result_block[i, j] = max(array)  # Максимальное число
                result_block[j, i] = result_block[i, j]  # Матрица симметричная

        mean = result_block[numpy.triu_indices(block.shape[0], k=1)].mean()
        std = result_block[numpy.triu_indices(block.shape[0], k=1)].std()
        result_block = (result_block - mean) / std
        result_block[numpy.diag_indices(result_block.shape[0])] = 0
        result.append(result_block)

    result = numpy.mean(result, axis=0)
    numpy.savetxt(file_out, result)


def granger_causality(file_in, file_out, lag=3):
    """
    Вычисление granger causality.

    :param file_in: Файл временных рядов регионов.
    :param file_out: Файл ребер.
    :param lag: Сдвиг.
    :return:
    """
    with open(file_in, 'rb') as file:
        data = pickle_load(file)
        data = [block.to_numpy() for block in data]

    result = []
    for count, block in enumerate(data):
        result_block = numpy.zeros((block.shape[0], block.shape[0]))
        for i in range(block.shape[0]):
            for j in range(i + 1, block.shape[0]):
                if (380 * i + j) % 1000 == 0:
                    print(count, 380 * i + j)
                block_ = pandas.DataFrame({'x': block[i, :], 'y': block[j, :]})  # влияние i на j
                gc = grangercausalitytests(block_, maxlag=lag, verbose=False)
                result_block[i, j] = max([gc[k][0]["ssr_ftest"][0] for k in range(1, lag)])  # Максимальное число

                block_ = pandas.DataFrame({'x': block[j, :], 'y': block[i, :]})  # влияние j на i
                gc = grangercausalitytests(block_, maxlag=lag, verbose=False)
                result_block[j, i] = max([gc[k][0]["ssr_ftest"][0] for k in range(1, lag)])  # Максимальное число

        mean = numpy.hstack((result_block[numpy.triu_indices(block.shape[0], k=1)],
                             result_block[numpy.tril_indices(block.shape[0], k=1)])).mean()
        std = numpy.hstack((result_block[numpy.triu_indices(block.shape[0], k=1)],
                            result_block[numpy.tril_indices(block.shape[0], k=1)])).std()
        result_block = (result_block - mean) / std
        result_block[numpy.diag_indices(result_block.shape[0])] = 0
        result.append(result_block)

    result = numpy.mean(result, axis=0)
    numpy.savetxt(file_out, result)


def transfer_entropy(file_in, file_out, lag=0, n_bins=4, strategy='quantile'):
    """
    Вычисление transfer entropy.

    :param file_in: Файл временных рядов регионов.
    :param file_out: Файл ребер.
    :param lag: История энтропии.
    :param n_bins: Число кластеров дискретизации.
    :param strategy: Стратегия дискретизации.
    :return:
    """
    with open(file_in, 'rb') as file:
        data = pickle_load(file)
        data = [block.to_numpy() for block in data]

    result = []
    for count, block in enumerate(data):
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        for i in range(block.shape[0]):
            block[i, :] = discretizer.fit_transform(block[i, :].reshape(-1, 1)).flatten()

        result_block = numpy.zeros((block.shape[0], block.shape[0]))
        for i in range(block.shape[0]):
            for j in range(i + 1, block.shape[0]):
                if (380 * i + j) % 1000 == 0:
                    print(count, 380 * i + j)

                array = []
                for k in range(1, lag + 1):
                    array.append(mutual_info_score(block[i, :-k], block[j, k:]) -   # i после j
                                 mutual_info_score(block[i, :-k], block[i, k:]))
                result_block[j, i] = max(array)

                array = []
                for k in range(1, lag + 1):
                    array.append(mutual_info_score(block[j, :-k], block[i, k:]) -  # i до j
                                 mutual_info_score(block[j, :-k], block[j, k:]))
                result_block[i, j] = max(array)

        mean = numpy.hstack((result_block[numpy.triu_indices(block.shape[0], k=1)],
                             result_block[numpy.tril_indices(block.shape[0], k=1)])).mean()
        std = numpy.hstack((result_block[numpy.triu_indices(block.shape[0], k=1)],
                            result_block[numpy.tril_indices(block.shape[0], k=1)])).std()
        result_block = (result_block - mean) / std
        result_block[numpy.diag_indices(result_block.shape[0])] = 0
        result.append(result_block)

    result = numpy.mean(result, axis=0)
    numpy.savetxt(file_out, result)


def mp(files_in, files_out, func, n_jobs=0, *args):
    """
    Параллельное вычисление весов ребер.

    :param files_in: Файлы временных рядов регионов.
    :param files_out: Файлы ребер.
    :param func: Функция вычисления связности.
    :param n_jobs: Число ядер.
    :return:
    """
    if n_jobs == 0:
        # n_jobs = multiprocessing.cpu_count() - 1
        n_jobs = 5

    task_list = [(files_in[i], files_out[i], *args) for i in range(len(files_in))]
    with multiprocessing.Pool(n_jobs) as pool:
        pool.starmap(func, task_list)
