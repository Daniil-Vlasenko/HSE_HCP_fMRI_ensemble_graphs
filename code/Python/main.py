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
    edge1, edge2 = paths.data_preparation(paths.files_mean_pearson_cross_correlation_wm, 'LR',
                                          paths.folds_gnn['fold0']['test'])
    vertex1, vertex2 = paths.data_preparation(paths.files_mean_vertices_wm, 'LR',
                                              paths.folds_gnn['fold0']['test'])
    # edges.models_learning([edge1, edge2], [vertex1, vertex2], paths.files_mean_models_wm)
    edges.models_calculation([edge1, edge2], [vertex1, vertex2], paths.files_mean_models_wm, paths.folders_mean_ensemble_edges_test_wm[0])