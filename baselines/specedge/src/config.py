import os
import torch
import util

class _ConfigMeta(type):
    _initialized = False

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        return cls

    def __getattr__(cls, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        if not cls._initialized:
            cls._initialize()
        if name in cls.__dict__:
            return cls.__dict__[name]
        else:
            raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    def __setattr__(cls, name, value):
        super().__setattr__(name, value)

    def _initialize(cls):
        raise NotImplementedError('Subclasses must implement _initialize()')

    def _from_env(cls, key: str):
        value = os.getenv(key)
        if value is None or value == 'null':
            raise ValueError(f"Environment variable '{key}' is not set")
        return value

    def reset(cls):
        cls._initialized = False
        cls._initialize()

class SpecEdgeClientConfig(metaclass=_ConfigMeta):

    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env('SPECEDGE_RESULT_PATH')
        cls.exp_name = cls._from_env('SPECEDGE_EXP_NAME')
        cls.process_name = cls._from_env('SPECEDGE_PROCESS_NAME')
        cls.seed = int(cls._from_env('SPECEDGE_SEED'))
        cls.optimization = int(cls._from_env('SPECEDGE_OPTIMIZATION'))
        cls.max_len = int(cls._from_env('SPECEDGE_MAX_LEN'))
        cls.draft_model = cls._from_env('SPECEDGE_DRAFT_MODEL')
        cls.device = torch.device(cls._from_env('SPECEDGE_DEVICE'))
        cls.dtype = util.convert_dtype(cls._from_env('SPECEDGE_DTYPE'))
        cls.reasoning = cls._from_env('SPECEDGE_REASONING') == 'True'
        cls.dataset = cls._from_env('SPECEDGE_DATASET')
        cls.max_n_beams = int(cls._from_env('SPECEDGE_MAX_N_BEAMS'))
        cls.max_beam_len = int(cls._from_env('SPECEDGE_MAX_BEAM_LEN'))
        cls.max_branch_width = int(cls._from_env('SPECEDGE_MAX_BRANCH_WIDTH'))
        cls.max_budget = int(cls._from_env('SPECEDGE_MAX_BUDGET'))
        cls.proactive_type = cls._from_env('SPECEDGE_PROACTIVE_TYPE')
        cls.proactive_max_n_beams = int(cls._from_env('SPECEDGE_PROACTIVE_MAX_N_BEAMS'))
        cls.proactive_max_beam_len = int(cls._from_env('SPECEDGE_PROACTIVE_MAX_BEAM_LEN'))
        cls.proactive_max_branch_width = int(cls._from_env('SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH'))
        cls.proactive_max_budget = int(cls._from_env('SPECEDGE_PROACTIVE_MAX_BUDGET'))
        cls.max_new_tokens = int(cls._from_env('SPECEDGE_MAX_NEW_TOKENS'))
        cls.max_request_num = int(cls._from_env('SPECEDGE_MAX_REQUEST_NUM'))
        cls.req_offset = int(cls._from_env('SPECEDGE_REQ_OFFSET'))
        cls.sample_req_cnt = int(cls._from_env('SPECEDGE_SAMPLE_REQ_CNT'))
        cls.host = cls._from_env('SPECEDGE_HOST')
        cls.client_idx = int(cls._from_env('SPECEDGE_CLIENT_IDX'))
        cls._initialized = True

class SpecEdgeBatchServerConfig(metaclass=_ConfigMeta):

    @classmethod
    def _initialize(cls):
        cls.result_path = cls._from_env('SPECEDGE_RESULT_PATH')
        cls.exp_name = cls._from_env('SPECEDGE_EXP_NAME')
        cls.process_name = cls._from_env('SPECEDGE_PROCESS_NAME')
        cls.seed = int(cls._from_env('SPECEDGE_SEED'))
        cls.max_len = int(cls._from_env('SPECEDGE_MAX_LEN'))
        cls.batch_type = cls._from_env('SPECEDGE_BATCH_TYPE')
        cls.dataset = cls._from_env('SPECEDGE_DATASET')
        cls.sample_req_cnt = int(cls._from_env('SPECEDGE_SAMPLE_REQ_CNT'))
        cls.req_offset = int(cls._from_env('SPECEDGE_REQ_OFFSET'))
        cls.target_model = cls._from_env('SPECEDGE_TARGET_MODEL')
        cls.device = torch.device(cls._from_env('SPECEDGE_SERVER_DEVICE'))
        cls.dtype = util.convert_dtype(cls._from_env('SPECEDGE_DTYPE'))
        cls.temperature = float(cls._from_env('SPECEDGE_TEMPERATURE'))
        cls.max_batch_size = int(cls._from_env('SPECEDGE_MAX_BATCH_SIZE'))
        cls.max_n_beams = int(cls._from_env('SPECEDGE_MAX_N_BEAMS'))
        cls.max_budget = int(cls._from_env('SPECEDGE_MAX_BUDGET'))
        cls.num_clients = int(cls._from_env('SPECEDGE_NUM_CLIENTS'))
        cls.cache_prefill = cls._from_env('SPECEDGE_CACHE_PREFILL') == 'True'
        cls._initialized = True
