"""
Ниже многие пути вычисляются относительно предобработанных данных HCP, так что если папки с исходными данными пусты,
то пусты будут некоторые массивы путей.
"""
from glob import glob
import numpy
from sklearn.model_selection import KFold

########## ДАННЫЕ ФМРТ ##################################################
# Папки парцеллированными данными, mean
folder_parcell_mean_wm = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_mean/WM"
folder_parcell_mean_gambling = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_mean/GAMBLING"
folder_parcell_mean_motor = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_mean/MOTOR"
folder_parcell_mean_language = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_mean/LANGUAGE"
folder_parcell_mean_social = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_mean/SOCIAL"
folder_parcell_mean_relational = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_mean/RELATIONAL"
folder_parcell_mean_emotion = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_mean/EMOTION"

# Файлы с парцеллированными данными, mean
## После генерации
files_parcell_mean_wm_ = sorted([f"{folder_parcell_mean_wm}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_mean_wm}/*")])
files_parcell_mean_gambling_ = sorted([f"{folder_parcell_mean_gambling}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_mean_gambling}/*")])
files_parcell_mean_motor_ = sorted([f"{folder_parcell_mean_motor}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_mean_motor}/*")])
files_parcell_mean_language_ = sorted([f"{folder_parcell_mean_language}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_mean_language}/*")])
files_parcell_mean_social_ = sorted([f"{folder_parcell_mean_social}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_mean_social}/*")])
files_parcell_mean_relational_ = sorted([f"{folder_parcell_mean_relational}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_mean_relational}/*")])
files_parcell_mean_emotion_ = sorted([f"{folder_parcell_mean_emotion}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_mean_emotion}/*")])

# Папки парцеллированными данными, pca
folder_parcell_pca_wm = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_pca/WM"
folder_parcell_pca_gambling = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_pca/GAMBLING"
folder_parcell_pca_motor = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_pca/MOTOR"
folder_parcell_pca_language = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_pca/LANGUAGE"
folder_parcell_pca_social = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_pca/SOCIAL"
folder_parcell_pca_relational = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_pca/RELATIONAL"
folder_parcell_pca_emotion = "../../../HCE_HCP_fMRI_graphs/processed_data/parcell_pca/EMOTION"

# Файлы с парцеллированными данными, pca
## После генерации
files_parcell_pca_wm_ = sorted([f"{folder_parcell_pca_wm}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_pca_wm}/*")])
files_parcell_pca_gambling_ = sorted([f"{folder_parcell_pca_gambling}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_pca_gambling}/*")])
files_parcell_pca_motor_ = sorted([f"{folder_parcell_pca_motor}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_pca_motor}/*")])
files_parcell_pca_language_ = sorted([f"{folder_parcell_pca_language}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_pca_language}/*")])
files_parcell_pca_social_ = sorted([f"{folder_parcell_pca_social}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_pca_social}/*")])
files_parcell_pca_relational_ = sorted([f"{folder_parcell_pca_relational}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_pca_relational}/*")])
files_parcell_pca_emotion_ = sorted([f"{folder_parcell_pca_emotion}/{file.split('/')[-1]}" for file in glob(f"{folder_parcell_pca_emotion}/*")])

# Папки с разделенными блоками, mean
folder_split_mean_wm = "../../../HCE_HCP_fMRI_graphs/processed_data/split_mean/WM"
folder_split_mean_gambling = "../../../HCE_HCP_fMRI_graphs/processed_data/split_mean/GAMBLING"
folder_split_mean_motor = "../../../HCE_HCP_fMRI_graphs/processed_data/split_mean/MOTOR"
folder_split_mean_language = "../../../HCE_HCP_fMRI_graphs/processed_data/split_mean/LANGUAGE"
folder_split_mean_social = "../../../HCE_HCP_fMRI_graphs/processed_data/split_mean/SOCIAL"
folder_split_mean_relational = "../../../HCE_HCP_fMRI_graphs/processed_data/split_mean/RELATIONAL"
folder_split_mean_emotion = "../../../HCE_HCP_fMRI_graphs/processed_data/split_mean/EMOTION"

# Файлы с разделенными блоками, mean
files_split_mean_wm = [f"{folder_split_mean_wm}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_mean_wm_ for type in ["0bk", "2bk"]]
files_split_mean_gambling = [f"{folder_split_mean_gambling}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_mean_gambling_ for type in ["win", "loss"]]
files_split_mean_motor = [f"{folder_split_mean_motor}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_mean_motor_ for type in ["l", "r"]]
files_split_mean_language = [f"{folder_split_mean_language}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_mean_language_ for type in ["story", "math"]]
files_split_mean_social = [f"{folder_split_mean_social}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_mean_social_ for type in ["rnd", "mental"]]
files_split_mean_relational = [f"{folder_split_mean_relational}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_mean_relational_ for type in ["relation", "match"]]
files_split_mean_emotion = [f"{folder_split_mean_emotion}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_mean_emotion_ for type in ["neut", "fear"]]

# Папки с разделенными блоками, pca
folder_split_pca_wm = "../../../HCE_HCP_fMRI_graphs/processed_data/split_pca/WM"
folder_split_pca_gambling = "../../../HCE_HCP_fMRI_graphs/processed_data/split_pca/GAMBLING"
folder_split_pca_motor = "../../../HCE_HCP_fMRI_graphs/processed_data/split_pca/MOTOR"
folder_split_pca_language = "../../../HCE_HCP_fMRI_graphs/processed_data/split_pca/LANGUAGE"
folder_split_pca_social = "../../../HCE_HCP_fMRI_graphs/processed_data/split_pca/SOCIAL"
folder_split_pca_relational = "../../../HCE_HCP_fMRI_graphs/processed_data/split_pca/RELATIONAL"
folder_split_pca_emotion = "../../../HCE_HCP_fMRI_graphs/processed_data/split_pca/EMOTION"

# Файлы с разделенными блоками, pca
files_split_pca_wm = [f"{folder_split_pca_wm}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_pca_wm_ for type in ["0bk", "2bk"]]
files_split_pca_gambling = [f"{folder_split_pca_gambling}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_pca_gambling_ for type in ["win", "loss"]]
files_split_pca_motor = [f"{folder_split_pca_motor}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_pca_motor_ for type in ["l", "r"]]
files_split_pca_language = [f"{folder_split_pca_language}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_pca_language_ for type in ["story", "math"]]
files_split_pca_social = [f"{folder_split_pca_social}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_pca_social_ for type in ["rnd", "mental"]]
files_split_pca_relational = [f"{folder_split_pca_relational}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_pca_relational_ for type in ["relation", "match"]]
files_split_pca_emotion = [f"{folder_split_pca_emotion}/{file.split('/')[-1][:-4]}_{type}.pickle" for file in files_parcell_pca_emotion_ for type in ["neut", "fear"]]
########## ДАННЫЕ ФМРТ ##################################################

########## РЕБРА ГРАФОВ ##################################################
# Папки с pearson correlation, mean
folder_mean_pearson_cross_correlation_wm = "../../../HCE_HCP_fMRI_graphs/graphs/edges/mean/pearson_cross_correlation/WM"
folder_mean_pearson_cross_correlation_gambling = "../../../HCE_HCP_fMRI_graphs/graphs/edges/mean/pearson_cross_correlation/GAMBLING"
folder_mean_pearson_cross_correlation_motor = "../../../HCE_HCP_fMRI_graphs/graphs/edges/mean/pearson_cross_correlation/MOTOR"
folder_mean_pearson_cross_correlation_language = "../../../HCE_HCP_fMRI_graphs/graphs/edges/mean/pearson_cross_correlation/LANGUAGE"
folder_mean_pearson_cross_correlation_social = "../../../HCE_HCP_fMRI_graphs/graphs/edges/mean/pearson_cross_correlation/SOCIAL"
folder_mean_pearson_cross_correlation_relational = "../../../HCE_HCP_fMRI_graphs/graphs/edges/mean/pearson_cross_correlation/RELATIONAL"
folder_mean_pearson_cross_correlation_emotion = "../../../HCE_HCP_fMRI_graphs/graphs/edges/mean/pearson_cross_correlation/EMOTION"

# Файлы с pearson correlation, mean
files_mean_pearson_cross_correlation_wm = [f'{folder_mean_pearson_cross_correlation_wm}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_wm]
files_mean_pearson_cross_correlation_gambling = [f'{folder_mean_pearson_cross_correlation_gambling}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_gambling]
files_mean_pearson_cross_correlation_motor = [f'{folder_mean_pearson_cross_correlation_motor}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_motor]
files_mean_pearson_cross_correlation_language = [f'{folder_mean_pearson_cross_correlation_language}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_language]
files_mean_pearson_cross_correlation_social = [f'{folder_mean_pearson_cross_correlation_social}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_social]
files_mean_pearson_cross_correlation_relational = [f'{folder_mean_pearson_cross_correlation_relational}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_relational]
files_mean_pearson_cross_correlation_emotion = [f'{folder_mean_pearson_cross_correlation_emotion}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_emotion]

# Папки с pearson correlation, pca
folder_pca_pearson_cross_correlation_wm = "../../../HCE_HCP_fMRI_graphs/graphs/edges/pca/pearson_cross_correlation/WM"
folder_pca_pearson_cross_correlation_gambling = "../../../HCE_HCP_fMRI_graphs/graphs/edges/pca/pearson_cross_correlation/GAMBLING"
folder_pca_pearson_cross_correlation_motor = "../../../HCE_HCP_fMRI_graphs/graphs/edges/pca/pearson_cross_correlation/MOTOR"
folder_pca_pearson_cross_correlation_language = "../../../HCE_HCP_fMRI_graphs/graphs/edges/pca/pearson_cross_correlation/LANGUAGE"
folder_pca_pearson_cross_correlation_social = "../../../HCE_HCP_fMRI_graphs/graphs/edges/pca/pearson_cross_correlation/SOCIAL"
folder_pca_pearson_cross_correlation_relational = "../../../HCE_HCP_fMRI_graphs/graphs/edges/pca/pearson_cross_correlation/RELATIONAL"
folder_pca_pearson_cross_correlation_emotion = "../../../HCE_HCP_fMRI_graphs/graphs/edges/pca/pearson_cross_correlation/EMOTION"

# Файлы с pearson correlation, pca
files_pca_pearson_cross_correlation_wm = [f'{folder_pca_pearson_cross_correlation_wm}/{file.split("/")[-1][:-6]}npy' for file in files_split_pca_wm]
files_pca_pearson_cross_correlation_gambling = [f'{folder_pca_pearson_cross_correlation_gambling}/{file.split("/")[-1][:-6]}npy' for file in files_split_pca_gambling]
files_pca_pearson_cross_correlation_motor = [f'{folder_pca_pearson_cross_correlation_motor}/{file.split("/")[-1][:-6]}npy' for file in files_split_pca_motor]
files_pca_pearson_cross_correlation_language = [f'{folder_pca_pearson_cross_correlation_language}/{file.split("/")[-1][:-6]}npy' for file in files_split_pca_language]
files_pca_pearson_cross_correlation_social = [f'{folder_pca_pearson_cross_correlation_social}/{file.split("/")[-1][:-6]}npy' for file in files_split_pca_social]
files_pca_pearson_cross_correlation_relational = [f'{folder_pca_pearson_cross_correlation_relational}/{file.split("/")[-1][:-6]}npy' for file in files_split_pca_relational]
files_pca_pearson_cross_correlation_emotion = [f'{folder_pca_pearson_cross_correlation_emotion}/{file.split("/")[-1][:-6]}npy' for file in files_split_pca_emotion]
########## РЕБРА ГРАФОВ ##################################################

########## ВЕРШИНЫ ГРАФОВ ##################################################
# Папки значений вершин, mean
folder_mean_vertices_wm = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/mean/WM"
folder_mean_vertices_gambling = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/mean/GAMBLING"
folder_mean_vertices_motor = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/mean/MOTOR"
folder_mean_vertices_language = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/mean/LANGUAGE"
folder_mean_vertices_social = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/mean/SOCIAL"
folder_mean_vertices_relational = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/mean/RELATIONAL"
folder_mean_vertices_emotion = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/mean/EMOTION"

# Файлы значений вершин, mean
files_mean_vertices_wm = [f'{folder_mean_vertices_wm}/{file.split("/")[-1][:-6]}pickle' for file in files_split_mean_wm]
files_mean_vertices_gambling = [f'{folder_mean_vertices_gambling}/{file.split("/")[-1][:-6]}pickle' for file in files_split_mean_gambling]
files_mean_vertices_motor = [f'{folder_mean_vertices_motor}/{file.split("/")[-1][:-6]}pickle' for file in files_split_mean_motor]
files_mean_vertices_language = [f'{folder_mean_vertices_language}/{file.split("/")[-1][:-6]}pickle' for file in files_split_mean_language]
files_mean_vertices_social = [f'{folder_mean_vertices_social}/{file.split("/")[-1][:-6]}pickle' for file in files_split_mean_social]
files_mean_vertices_relational = [f'{folder_mean_vertices_relational}/{file.split("/")[-1][:-6]}pickle' for file in files_split_mean_relational]
files_mean_vertices_emotion = [f'{folder_mean_vertices_emotion}/{file.split("/")[-1][:-6]}pickle' for file in files_split_mean_emotion]

# Папки значений вершин, pca
folder_pca_vertices_wm = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/pca/WM"
folder_pca_vertices_gambling = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/pca/GAMBLING"
folder_pca_vertices_motor = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/pca/MOTOR"
folder_pca_vertices_language = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/pca/LANGUAGE"
folder_pca_vertices_social = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/pca/SOCIAL"
folder_pca_vertices_relational = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/pca/RELATIONAL"
folder_pca_vertices_emotion = "../../../HCE_HCP_fMRI_graphs/graphs/vertices/pca/EMOTION"

# Файлы значений вершин, pca
files_pca_vertices_wm = [f'{folder_pca_vertices_wm}/{file.split("/")[-1][:-6]}pickle' for file in files_split_pca_wm]
files_pca_vertices_gambling = [f'{folder_pca_vertices_gambling}/{file.split("/")[-1][:-6]}pickle' for file in files_split_pca_gambling]
files_pca_vertices_motor = [f'{folder_pca_vertices_motor}/{file.split("/")[-1][:-6]}pickle' for file in files_split_pca_motor]
files_pca_vertices_language = [f'{folder_pca_vertices_language}/{file.split("/")[-1][:-6]}pickle' for file in files_split_pca_language]
files_pca_vertices_social = [f'{folder_pca_vertices_social}/{file.split("/")[-1][:-6]}pickle' for file in files_split_pca_social]
files_pca_vertices_relational = [f'{folder_pca_vertices_relational}/{file.split("/")[-1][:-6]}pickle' for file in files_split_pca_relational]
files_pca_vertices_emotion = [f'{folder_pca_vertices_emotion}/{file.split("/")[-1][:-6]}pickle' for file in files_split_pca_emotion]
########## ВЕРШИНЫ ГРАФОВ ##################################################

########## АНСАМБЛИ ##################################################
# Папки моделей, mean
folder_mean_models_wm = "../../graphs/mean/models/WM"
folder_mean_models_gambling = "../../graphs/mean/models/GAMBLING"
folder_mean_models_motor = "../../graphs/mean/models/MOTOR"
folder_mean_models_language = "../../graphs/mean/models/LANGUAGE"
folder_mean_models_social = "../../graphs/mean/models/SOCIAL"
folder_mean_models_relational = "../../graphs/mean/models/RELATIONAL"
folder_mean_models_emotion = "../../graphs/mean/models/EMOTION"

# Файлы моделей, mean
files_mean_models_wm = [f'{folder_mean_models_wm}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_mean_models_gambling = [f'{folder_mean_models_gambling}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_mean_models_motor = [f'{folder_mean_models_motor}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_mean_models_language = [f'{folder_mean_models_language}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_mean_models_social = [f'{folder_mean_models_social}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_mean_models_relational = [f'{folder_mean_models_relational}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_mean_models_emotion = [f'{folder_mean_models_emotion}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]

# Папки моделей, pca
folder_pca_models_wm = "../../graphs/pca/models/WM"
folder_pca_models_gambling = "../../graphs/pca/models/GAMBLING"
folder_pca_models_motor = "../../graphs/pca/models/MOTOR"
folder_pca_models_language = "../../graphs/pca/models/LANGUAGE"
folder_pca_models_social = "../../graphs/pca/models/SOCIAL"
folder_pca_models_relational = "../../graphs/pca/models/RELATIONAL"
folder_pca_models_emotion = "../../graphs/pca/models/EMOTION"

# Файлы моделей, pca
files_pca_models_wm = [f'{folder_pca_models_wm}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_pca_models_gambling = [f'{folder_pca_models_gambling}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_pca_models_motor = [f'{folder_pca_models_motor}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_pca_models_language = [f'{folder_pca_models_language}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_pca_models_social = [f'{folder_pca_models_social}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_pca_models_relational = [f'{folder_pca_models_relational}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]
files_pca_models_emotion = [f'{folder_pca_models_emotion}/{i}_{j}.pickle' for i in range(379) for j in range(i + 1, 379)]

# Папка с ансамблевыми ребрами, mean
folders_mean_ensemble_edges_test = [f'../../graphs/mean/ensemble_edges/fold{i}/test' for i in range(4)]
folders_mean_ensemble_edges_train = [[f'../../graphs/mean/ensemble_edges/fold{i}/train/fold{j}' for j in range(3)] for i in range(4)]

# Папка с ансамблевыми ребрами, pca
folders_pca_ensemble_edges_test = [f'../../graphs/pca/ensemble_edges/fold{i}/test' for i in range(4)]
folders_pca_ensemble_edges_train = [[f'../../graphs/pca/ensemble_edges/fold{i}/train/fold{j}' for j in range(3)] for i in range(4)]

# Делим двухуровнево выборку
folds_gnn = {'fold0': {'train': None, 'test': None},
             'fold1': {'train': None, 'test': None},
             'fold2': {'train': None, 'test': None},
             'fold3': {'train': None, 'test': None}}

folds_ensemble = {'fold0': {'fold0': {'train': None, 'test': None},
                            'fold1': {'train': None, 'test': None},
                            'fold2': {'train': None, 'test': None}},
                  'fold1': {'fold0': {'train': None, 'test': None},
                            'fold1': {'train': None, 'test': None},
                            'fold2': {'train': None, 'test': None}},
                  'fold2': {'fold0': {'train': None, 'test': None},
                            'fold1': {'train': None, 'test': None},
                            'fold2': {'train': None, 'test': None}},
                  'fold3': {'fold0': {'train': None, 'test': None},
                            'fold1': {'train': None, 'test': None},
                            'fold2': {'train': None, 'test': None}}}

people = numpy.array([int(index.split('/')[-1][:6]) for index in files_parcell_mean_wm_ if 'LR' in index])
kf_gnn = KFold(n_splits=4, shuffle=True, random_state=1)
for fold_gnn, (train_index_gnn, test_index_gnn) in enumerate(kf_gnn.split(people)):
    people_train_gnn, people_test_gnn = people[train_index_gnn], people[test_index_gnn]
    folds_gnn[f'fold{fold_gnn}']['train'] = people_train_gnn
    folds_gnn[f'fold{fold_gnn}']['test'] = people_test_gnn

    kf_ensemble = KFold(n_splits=3, shuffle=True, random_state=1)
    for fold_ensemble, (train_index_ensemble, test_index_ensemble) in enumerate(kf_ensemble.split(people_train_gnn)):
        people_train_ensemble, people_test_ensemble = people_train_gnn[train_index_ensemble], people_train_gnn[test_index_ensemble]
        folds_ensemble[f'fold{fold_gnn}'][f'fold{fold_ensemble}']['train'] = people_train_ensemble
        folds_ensemble[f'fold{fold_gnn}'][f'fold{fold_ensemble}']['test'] = people_test_ensemble




# Файлы ансамблевых графов, mean
files_mean_ensgraphs_wm = [f'{folder_mean_pearson_cross_correlation_wm}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_wm]
files_mean_pearson_cross_correlation_gambling = [f'{folder_mean_pearson_cross_correlation_gambling}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_gambling]
files_mean_pearson_cross_correlation_motor = [f'{folder_mean_pearson_cross_correlation_motor}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_motor]
files_mean_pearson_cross_correlation_language = [f'{folder_mean_pearson_cross_correlation_language}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_language]
files_mean_pearson_cross_correlation_social = [f'{folder_mean_pearson_cross_correlation_social}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_social]
files_mean_pearson_cross_correlation_relational = [f'{folder_mean_pearson_cross_correlation_relational}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_relational]
files_mean_pearson_cross_correlation_emotion = [f'{folder_mean_pearson_cross_correlation_emotion}/{file.split("/")[-1][:-6]}npy' for file in files_split_mean_emotion]
########## АНСАМБЛИ ##################################################



