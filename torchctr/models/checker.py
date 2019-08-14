#!/usr/bin/env python
# encoding: utf-8

MODEL_PARAMS = {
    "LogisticRegression": {"feature_dims": None},
    "FactorizationMachine": {"feature_dims": None, "embed_dim": 4},
    "FieldAwareFactorizationMachine": {"feature_dims": None, "embed_dim": 4},
    "WideAndDeepModel": {"feature_dims": None, "embed_dim": 4, "hidden_dims": [10,10,10]},
    "DeepFactorizationMachine": {"feature_dims": None, "embed_dim": 4, "hidden_dims": [10,10,10]},
    "NeuralFactorizationMachine": {"feature_dims": None, "embed_dim": 4, "hidden_dims": [10,10,10]},
    "FieldAwareNeuralFactorizationMachine": {}
}


class Checker(object):
    """
    Checker for model arguments

    """

    @classmethod
    def model_param_check(cls, init_model):
        def wrapper(self, *args, **kwargs):
            model_name = str(self)
            if model_name not in MODEL_PARAMS.keys():
                raise ValueError("Model not supported")
            rebuild_params = cls.model_check(model_name, kwargs)
            return init_model(self, *args, **rebuild_params)
        return wrapper

    @staticmethod
    def model_check(model_name, kwargs):
        default_params = MODEL_PARAMS[model_name]
        rebuild_params = default_params.copy()

        for k, v in default_params.items():
            if k not in kwargs.keys():
                if default_params[k] is None:
                    raise ValueError("| ERROR | {} must be specified".format(k))
                else:
                    print("| WARNING | {} should be specified, otherwise we'll use default value {}".format(k, default_params[k]))
            else:
                rebuild_params[k] = kwargs[k]

        for k in kwargs.keys():
            if k not in default_params.keys():
                print("| WARNING | we dont know {}, we'll ignore this".format(k))

        return rebuild_params
