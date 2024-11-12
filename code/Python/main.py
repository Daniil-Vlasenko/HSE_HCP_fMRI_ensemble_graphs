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
    # print('wm')
    # edges.cross_validation(files_in1=paths.files_mean_pearson_cross_correlation_wm,
    #                        files_in2=paths.files_mean_vertices_wm,
    #                        encoding_type=encoding_type, models_files=paths.files_mean_models_wm,
    #                        folder_out1=paths.folders_mean_ensemble_edges_train_wm,
    #                        folder_out2=paths.folders_mean_ensemble_edges_test_wm)
    # print('\ngambling')
    # edges.cross_validation(files_in1=paths.files_mean_pearson_cross_correlation_gambling,
    #                        files_in2=paths.files_mean_vertices_gambling,
    #                        encoding_type=encoding_type, models_files=paths.files_mean_models_gambling,
    #                        folder_out1=paths.folders_mean_ensemble_edges_train_gambling,
    #                        folder_out2=paths.folders_mean_ensemble_edges_test_gambling)
    # print('\nmotor')
    # edges.cross_validation(files_in1=paths.files_mean_pearson_cross_correlation_motor,
    #                        files_in2=paths.files_mean_vertices_motor,
    #                        encoding_type=encoding_type, models_files=paths.files_mean_models_motor,
    #                        folder_out1=paths.folders_mean_ensemble_edges_train_motor,
    #                        folder_out2=paths.folders_mean_ensemble_edges_test_motor)
    # print('\nlanguage')
    # edges.cross_validation(files_in1=paths.files_mean_pearson_cross_correlation_language,
    #                        files_in2=paths.files_mean_vertices_language,
    #                        encoding_type=encoding_type, models_files=paths.files_mean_models_language,
    #                        folder_out1=paths.folders_mean_ensemble_edges_train_language,
    #                        folder_out2=paths.folders_mean_ensemble_edges_test_language)
    # print('\nsocial')
    # edges.cross_validation(files_in1=paths.files_mean_pearson_cross_correlation_social,
    #                        files_in2=paths.files_mean_vertices_social,
    #                        encoding_type=encoding_type, models_files=paths.files_mean_models_social,
    #                        folder_out1=paths.folders_mean_ensemble_edges_train_social,
    #                        folder_out2=paths.folders_mean_ensemble_edges_test_social)
    # print('\nrelational')
    # edges.cross_validation(files_in1=paths.files_mean_pearson_cross_correlation_relational,
    #                        files_in2=paths.files_mean_vertices_relational,
    #                        encoding_type=encoding_type, models_files=paths.files_mean_models_relational,
    #                        folder_out1=paths.folders_mean_ensemble_edges_train_relational,
    #                        folder_out2=paths.folders_mean_ensemble_edges_test_relational)
    # print('\nemotion')
    # edges.cross_validation(files_in1=paths.files_mean_pearson_cross_correlation_emotion,
    #                        files_in2=paths.files_mean_vertices_emotion,
    #                        encoding_type=encoding_type, models_files=paths.files_mean_models_emotion,
    #                        folder_out1=paths.folders_mean_ensemble_edges_train_emotion,
    #                        folder_out2=paths.folders_mean_ensemble_edges_test_emotion)

    print('wm')
    edges.cross_validation(files_in1=paths.files_pca_pearson_cross_correlation_wm,
                           files_in2=paths.files_pca_vertices_wm,
                           encoding_type=encoding_type, models_files=paths.files_pca_models_wm,
                           folder_out1=paths.folders_pca_ensemble_edges_train_wm,
                           folder_out2=paths.folders_pca_ensemble_edges_test_wm)
    print('\ngambling')
    edges.cross_validation(files_in1=paths.files_pca_pearson_cross_correlation_gambling,
                           files_in2=paths.files_pca_vertices_gambling,
                           encoding_type=encoding_type, models_files=paths.files_pca_models_gambling,
                           folder_out1=paths.folders_pca_ensemble_edges_train_gambling,
                           folder_out2=paths.folders_pca_ensemble_edges_test_gambling)
    print('\nmotor')
    edges.cross_validation(files_in1=paths.files_pca_pearson_cross_correlation_motor,
                           files_in2=paths.files_pca_vertices_motor,
                           encoding_type=encoding_type, models_files=paths.files_pca_models_motor,
                           folder_out1=paths.folders_pca_ensemble_edges_train_motor,
                           folder_out2=paths.folders_pca_ensemble_edges_test_motor)
    print('\nlanguage')
    edges.cross_validation(files_in1=paths.files_pca_pearson_cross_correlation_language,
                           files_in2=paths.files_pca_vertices_language,
                           encoding_type=encoding_type, models_files=paths.files_pca_models_language,
                           folder_out1=paths.folders_pca_ensemble_edges_train_language,
                           folder_out2=paths.folders_pca_ensemble_edges_test_language)
    print('\nsocial')
    edges.cross_validation(files_in1=paths.files_pca_pearson_cross_correlation_social,
                           files_in2=paths.files_pca_vertices_social,
                           encoding_type=encoding_type, models_files=paths.files_pca_models_social,
                           folder_out1=paths.folders_pca_ensemble_edges_train_social,
                           folder_out2=paths.folders_pca_ensemble_edges_test_social)
    print('\nrelational')
    edges.cross_validation(files_in1=paths.files_pca_pearson_cross_correlation_relational,
                           files_in2=paths.files_pca_vertices_relational,
                           encoding_type=encoding_type, models_files=paths.files_pca_models_relational,
                           folder_out1=paths.folders_pca_ensemble_edges_train_relational,
                           folder_out2=paths.folders_pca_ensemble_edges_test_relational)
    print('\nemotion')
    edges.cross_validation(files_in1=paths.files_pca_pearson_cross_correlation_emotion,
                           files_in2=paths.files_pca_vertices_emotion,
                           encoding_type=encoding_type, models_files=paths.files_pca_models_emotion,
                           folder_out1=paths.folders_pca_ensemble_edges_train_emotion,
                           folder_out2=paths.folders_pca_ensemble_edges_test_emotion)
