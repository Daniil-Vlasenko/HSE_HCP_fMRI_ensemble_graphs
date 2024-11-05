import numpy
import time

import data_preparation as dp
import edges
import paths
import edges_


# if __name__ == '__main__':
    # Пересоздание папок перед скачиванием (так легче удалять файлы, сгенерированный ранее).
    # dp.recreate_folder(folder=paths. ...)

    # Распаковка данных после скачивания (Нужно запускать в отрыве от функций ниже,
    # иначе файлы не все будут использованы почему-то).

if __name__ == '__main__':
    paths.data_preparation(paths.files_mean_pearson_cross_correlation_wm, 'LR', paths.folds_gnn['fold0']['test'])
    edges.ensemble_classifier_learning()