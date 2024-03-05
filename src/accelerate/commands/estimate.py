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
from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from accelerate import init_empty_weights
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import (
    calculate_maximum_sizes,
    convert_bytes,
    is_timm_available,
    is_transformers_available,
)


if is_transformers_available():
    import transformers
    from transformers import AutoConfig, AutoModel

if is_timm_available():
    import timm


def verify_on_hub(repo: str, token: str = None):
    "Verifies that the model is on the hub and returns the model info."
    try:
        return model_info(repo, token=token)
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


def create_empty_model(model_name: str, library_name: str, trust_remote_code: bool = False, access_token: str = None):
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
        access_token (`str`, `optional`, defaults to `None`):
            The access token to use to access private or gated models on the Hub. (for use on the Gradio app)

    Returns:
        `torch.nn.Module`: The torch model that has been initialized on the `meta` device.

    """
    model_info = verify_on_hub(model_name, access_token)
    # Simplified errors
    if model_info == "gated":
        raise GatedRepoError(
            f"Repo for model `{model_name}` is gated. You must be authenticated to access it. Please run `huggingface-cli login`."
        )
    elif model_info == "repo":
        raise RepositoryNotFoundError(
            f"Repo for model `{model_name}` does not exist on the Hub. If you are trying to access a private repo,"
            " make sure you are authenticated via `huggingface-cli login` and have access."
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
        if model_info.config is None:
            raise RuntimeError(f"Tried to load `{model_name}` with `transformers` but it does not have any metadata.")

        auto_map = model_info.config.get("auto_map", False)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code, token=access_token)

        with init_empty_weights():
            # remote code could specify a specific `AutoModel` class in the `auto_map`
            constructor = AutoModel
            if isinstance(auto_map, dict):
                value = None
                for key in auto_map.keys():
                    if key.startswith("AutoModelFor"):
                        value = key
                        break
                if value is not None:
                    constructor = getattr(transformers, value)
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
    diff = 0

    def make_row(left_char, middle_char, right_char):
        return f"{left_char}{middle_char.join([in_between * n for n in column_widths])}{in_between * diff}{right_char}"

    separator = make_row("├", "┼", "┤")
    if len(title) > sum(column_widths):
        diff = abs(len(title) - len(separator))
        column_widths[-1] += diff

    # Update with diff
    separator = make_row("├", "┼", "┤")
    initial_rows = [
        make_row("┌", in_between, "┐"),
        f"{sep_char}{title.center(len(separator) - 2)}{sep_char}",
        make_row("├", "┬", "┤"),
    ]
    table = "\n".join(initial_rows) + "\n"
    column_widths[-1] += diff
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
        parser = CustomArgumentParser(description="Model size estimator for fitting a model onto CUDA memory.")

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

    title = f"Memory Usage for loading `{args.model_name}`"
    table = create_ascii_table(headers, data, title)
    print(table)


def main():
    parser = estimate_command_parser()
    args = parser.parse_args()
    estimate_command(args)


if __name__ == "__main__":
    main()
