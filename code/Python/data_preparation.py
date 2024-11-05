import multiprocessing
from glob import glob
from zipfile import ZipFile
from os import mkdir
from shutil import rmtree


def unzip_(folder, file, count):
    with ZipFile(file, "r") as zip_file:
        zip_file.extractall(folder)
        print(count, "file:", file)


def unzip(folder, n_jobs=0):
    """
    Распаковывает файлы в ту же папку, где они содержались.

    :param folder: Папка с оригинальными зип папками данных.
    :param n_jobs: Число параллельных процессов.
    :return:
    """
    # Делим вычисления между ядрами.
    files = glob(folder + "/*.zip")
    if n_jobs == 0:
        # n_jobs = multiprocessing.cpu_count() - 1
        n_jobs = 1

    task_list = [(folder, files[i], i) for i in range(len(files))]
    with multiprocessing.Pool(n_jobs) as pool:
        pool.starmap(unzip_, task_list)


def recreate_folder(folder):
    """
    Удаляет папку и создает ее заново.

    :param folder: Папка, которая будет пересоздана.
    :return:
    """
    rmtree(folder)
    mkdir(folder)
