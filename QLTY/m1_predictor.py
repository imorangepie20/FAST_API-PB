"""
M1 Predictor — 하위 호환 래퍼

실제 구현은 DBR.m1_predictor로 이동됨.
기존 `from QLTY import m1_predictor` 코드 호환을 위해 유지.
"""
from DBR.m1_predictor import *  # noqa: F401,F403
from DBR.m1_predictor import predict, is_loaded, M1_FEATURES  # noqa: F401
