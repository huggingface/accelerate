#!/usr/bin/env python

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

from accelerate import init_empty_weights
from accelerate.utils import (
    calculate_maximum_sizes,
    convert_bytes,
    is_huggingface_hub_available,
    is_timm_available,
    is_transformers_available,
)


if is_huggingface_hub_available():
    from huggingface_hub import model_info
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

if is_transformers_available():
    import transformers
    from transformers import AutoConfig, AutoModel

if is_timm_available():
    import timm


def verify_on_hub(repo: str):
    "Verifies that the model is on the hub and returns the model info."
    if not is_huggingface_hub_available():
        raise ImportError(
            "To use `accelerate estimate-memory`, the `huggingface_hub` library must be installed. Please run `pip install huggingface_hub` and try again."
        )
    try:
        return model_info(repo)
    except GatedRepoError:
        return "gated"
    except RepositoryNotFoundError:
        return "repo"


def check_has_model(error):
    """
    Checks what library spawned `error` when a model is not found
    """
    if is_timm_available() and isinstance(error, RuntimeError) and "Unknown model" in error.args[0]:
        return "timm"
    elif (
        is_transformers_available()
        and isinstance(error, OSError)
        and "does not appear to have a file named" in error.args[0]
    ):
        return "transformers"
    else:
        return "unknown"


def create_empty_model(model_name, library_name: str, trust_remote_code: bool = False):
    """
    Creates an empty model from its parent library on the `Hub` to calculate the overall memory consumption.

    Args:
        model_name (`str`):
            The model name on the Hub
        library_name (`str`):
            The library the model has an integration with, such as `transformers`. Will be used if `model_name` has no
            metadata on the Hub to determine the library.
        trust_remote_code (`bool`, `optional`, defaults to `False`):
            Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

    """
    model_info = verify_on_hub(model_name)
    # Simplified errors
    if model_info == "gated":
        raise GatedRepoError(f"Repo for model `{model_name}` is gated. You must be authenticated to access it.")
    elif model_info == "repo":
        raise RepositoryNotFoundError(
            f"Repo for model `{model_name}` does not exist on the Hub. If you are trying to access a private repo, make sure you are authenticated."
        )
    if library_name is None:
        library_name = getattr(model_info, "library_name", False)
        if not library_name:
            raise ValueError(
                f"Model `{model_name}` does not have any library metadata on the Hub, please manually pass in a `--library_name` to use (such as `transformers`)"
            )
    if library_name == "transformers":
        if not is_transformers_available():
            raise ImportError(
                f"To check `{model_name}`, `transformers` must be installed. Please install it via `pip install transformers`"
            )
        print(f"Loading pretrained config for `{model_name}` from `transformers`...")
        # As we just load in the `config` file and not the weights, we use `True`
        auto_map = model_info.config.get("auto_map", False)
        if auto_map and not trust_remote_code:
            raise ValueError(
                f"Loading {model_name} requires to execute some code in that repo, you can inspect the content of "
                f"the repository at https://hf.co/{model_name}. You can bypass this by passing "
                "`--trust_remote_code`"
            )

        try:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        except ValueError as e:
            if "trust_remote_code" in e.args[0]:
                e.args = (
                    f"Loading {model_name} requires to execute some code in that repo, you can inspect the content of "
                    f"the repository at https://hf.co/{model_name}. You can bypass this by passing "
                    "`--trust_remote_code`",
                )
            raise e
        with init_empty_weights():
            # remote code could specify a specific `AutoModel` class in the `auto_map`
            if isinstance(auto_map, dict):
                for key in auto_map.keys():
                    if key.startswith("AutoModelFor"):
                        value = key
                        break
                constructor = getattr(transformers, value)
            else:
                constructor = AutoModel
            model = constructor.from_config(config, trust_remote_code=trust_remote_code)
    elif library_name == "timm":
        if not is_timm_available():
            raise ImportError(
                f"To check `{model_name}`, `timm` must be installed. Please install it via `pip install timm`"
            )
        print(f"Loading pretrained config for `{model_name}` from `timm`...")
        with init_empty_weights():
            model = timm.create_model(model_name, pretrained=False)
    else:
        raise ValueError(
            f"Library `{library_name}` is not supported yet, please open an issue on GitHub for us to add support."
        )
    return model


def create_ascii_table(headers: list, rows: list, title: str):
    "Creates a pretty table from a list of rows, minimal version of `tabulate`."
    sep_char, in_between = "│", "─"
    column_widths = []
    for i in range(len(headers)):
        column_values = [row[i] for row in rows] + [headers[i]]
        max_column_width = max(len(value) for value in column_values)
        column_widths.append(max_column_width)

    formats = [f"%{column_widths[i]}s" for i in range(len(rows[0]))]

    pattern = f"{sep_char}{sep_char.join(formats)}{sep_char}"
    separator = f'├{"┼".join([in_between * n for n in column_widths])}┤'
    initial_rows = [
        f"┌{in_between * (len(separator)-2)}┐",
        f"{sep_char}{title.center(sum(column_widths) + len(column_widths)-1)}{sep_char}",
        f'├{"┬".join([in_between * n for n in column_widths])}┤',
    ]
    table = "\n".join(initial_rows) + "\n"
    centered_line = [text.center(column_widths[i]) for i, text in enumerate(headers)]
    table += f"{pattern % tuple(centered_line)}\n{separator}\n"
    for i, line in enumerate(rows):
        centered_line = [t.center(column_widths[i]) for i, t in enumerate(line)]
        table += f"{pattern % tuple(centered_line)}\n"
    table += f'└{"┴".join([in_between * n for n in column_widths])}┘'

    return table


def estimate_command_parser(subparsers=None):
    if subparsers is not None:
        parser = subparsers.add_parser("estimate-memory")
    else:
        parser = argparse.ArgumentParser(description="Model size estimator")

    parser.add_argument("model_name", type=str, help="The model name on the Hugging Face Hub.")
    parser.add_argument(
        "--library_name",
        type=str,
        help="The library the model has an integration with, such as `transformers`, needed only if this information is not stored on the Hub.",
        choices=["timm", "transformers"],
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["float32", "float16", "int8", "int4"],
        help="The dtypes to use for the model, must be one (or many) of `float32`, `float16`, `int8`, and `int4`",
        choices=["float32", "float16", "int8", "int4"],
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="""Whether or not to allow for custom models defined on the Hub in their own modeling files. This flag
                should only be used for repositories you trust and in which you have read the code, as it will execute
                code present on the Hub on your local machine.""",
    )

    if subparsers is not None:
        parser.set_defaults(func=estimate_command)
    return parser


def gather_data(args):
    "Creates an empty model and gathers the data for the sizes"
    try:
        model = create_empty_model(
            args.model_name, library_name=args.library_name, trust_remote_code=args.trust_remote_code
        )
    except (RuntimeError, OSError) as e:
        library = check_has_model(e)
        if library != "unknown":
            raise RuntimeError(
                f"Tried to load `{args.model_name}` with `{library}` but a possible model to load was not found inside the repo."
            )
        raise e

    total_size, largest_layer = calculate_maximum_sizes(model)

    data = []

    for dtype in args.dtypes:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        if dtype == "float16":
            dtype_total_size /= 2
            dtype_largest_layer /= 2
        elif dtype == "int8":
            dtype_total_size /= 4
            dtype_largest_layer /= 4
        elif dtype == "int4":
            dtype_total_size /= 8
            dtype_largest_layer /= 8
        dtype_training_size = dtype_total_size * 4
        data.append([dtype, dtype_largest_layer, dtype_total_size, dtype_training_size])
    return data


def estimate_command(args):
    data = gather_data(args)
    for row in data:
        for i, item in enumerate(row):
            if isinstance(item, (int, float)):
                row[i] = convert_bytes(item)

    headers = ["dtype", "Largest Layer", "Total Size", "Training using Adam"]

    title = f"Memory Usage for `{args.model_name}`"
    table = create_ascii_table(headers, data, title)
    print(table)


def main():
    parser = estimate_command_parser()
    args = parser.parse_args()
    estimate_command(args)


if __name__ == "__main__":
    main()