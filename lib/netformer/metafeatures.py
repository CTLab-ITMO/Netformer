import numpy
from pymfe.mfe import MFE

METAFEATURES = ("inst_to_attr, nr_class, nr_attr, attr_to_inst, skewness, kurtosis, cor, cov, attr_conc, class_conc, "
                "sparsity, gravity, skewness, class_ent, attr_ent, mut_inf, eq_num_attr, ns_ratio, f1, f2, tree_depth, "
                "leaves_branch, nodes_per_attr, leaves_per_class").split(", ")


def get_meta_features(parameters: numpy.array, target: numpy.array, meta_features: list[str] = None):
    if meta_features is None:
        meta_features = METAFEATURES
    meta_features = meta_features

    mfe = MFE(features=[*meta_features])
    mfe.fit(parameters, target)
    ft = mfe.extract()
    return ft[1]
