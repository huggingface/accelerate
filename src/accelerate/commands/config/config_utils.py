from accelerate.state import ComputeEnvironment, DistributedType, SageMakerDistributedType


def _ask_field(input_text, convert_value=None, default=None, error_message=None):
    ask_again = True
    while ask_again:
        result = input(input_text)
        try:
            if default is not None and len(result) == 0:
                return default
            return convert_value(result) if convert_value is not None else result
        except:
            if error_message is not None:
                print(error_message)


def _convert_compute_environment(value):
    value = int(value)
    return ComputeEnvironment(["CUSTOM_CLUSTER", "AMAZON_SAGEMAKER"][value])


def _convert_distributed_mode(value):
    value = int(value)
    return DistributedType(["NO", "MULTI_GPU", "TPU"][value])


def _convert_sagemaker_distributed_mode(value):
    value = int(value)
    return SageMakerDistributedType(["NO", "DATA_PARALLEL", "MODEL_PARALLEL"][value])


def _convert_yes_no_to_bool(value):
    return {"yes": True, "no": False}[value.lower()]
