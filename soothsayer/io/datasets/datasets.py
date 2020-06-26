# ==============================================================================
# Modules
# ==============================================================================
# Soothsayer
import soothsayer_utils as syu

functions_from_soothsayer_utils = [
'get_iris_data',
]


for function_name in functions_from_soothsayer_utils:
    globals()[function_name] = getattr(syu, function_name)