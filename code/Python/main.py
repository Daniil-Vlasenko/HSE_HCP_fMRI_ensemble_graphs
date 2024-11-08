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
    encoding_type = 'RL'
    # for

    edge1_tr, edge2_tr = paths.data_preparation(paths.files_mean_pearson_cross_correlation_wm,
                                                'LR', paths.folds_ensemble['fold0']['fold0']['train'])
    vertex1_tr, vertex2_tr = paths.data_preparation(paths.files_mean_vertices_wm,
                                                    'LR', paths.folds_ensemble['fold0']['fold0']['train'])
    dp.recreate_folder(folder=paths.folder_mean_models_wm)
    edges.models_learning([edge1_tr, edge2_tr], [vertex1_tr, vertex2_tr], paths.files_mean_models_wm)

    # построим трейн на основе трейна
    dp.recreate_folder(folder='/Users/daniilvlasenko/Programming/HSE_HCP_fMRI_ensemble_graphs/tmp/train_to_train')
    # edges.models_calculation([edge1_tr, edge2_tr], [vertex1_tr, vertex2_tr], paths.files_mean_models_wm,
    #                          '/Users/daniilvlasenko/Programming/HSE_HCP_fMRI_ensemble_graphs/tmp/train_to_train')

    # построим тест на основе трейна
    edge1_te, edge2_te = paths.data_preparation(paths.files_mean_pearson_cross_correlation_wm,
                                                'LR', paths.folds_ensemble['fold0']['fold0']['test'])
    vertex1_te, vertex2_te = paths.data_preparation(paths.files_mean_vertices_wm,
                                                    'LR', paths.folds_ensemble['fold0']['fold0']['test'])
    dp.recreate_folder(folder='/Users/daniilvlasenko/Programming/HSE_HCP_fMRI_ensemble_graphs/tmp/train_to_test')
    edges.models_calculation([edge1_te, edge2_te], [vertex1_te, vertex2_te], paths.files_mean_models_wm,
                             '/Users/daniilvlasenko/Programming/HSE_HCP_fMRI_ensemble_graphs/tmp/train_to_test')
