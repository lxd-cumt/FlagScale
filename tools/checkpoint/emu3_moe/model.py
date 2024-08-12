import time
from megatron.core.enums import ModelType

model_type = ModelType.encoder_or_decoder # Megatron's model_type


def get_hf_model(dtype, model_path=None, config=None):
    raise NotImplementedError()


def get_mg_model(dtype, pre_process, post_process):
    from flagscale.train.train_emu3 import model_provider
    s_time = time.time()
    model = model_provider(pre_process, post_process).to(dtype)
    print("> build megatron model elapsed time:", time.time() - s_time)
    return model
