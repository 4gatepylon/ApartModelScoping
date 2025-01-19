from __future__ import annotations
from pydantic import BaseModel
from typing import List, Optional, Tuple, Set, Dict, Any, Iterable
from pprint import pformat
from pathlib import Path
import re
import json
import tqdm
from scripts.datasets_eval import EvalDatasetsBuilder

"""
Provide methods to parse single generations files. Each generations file is a file
that corresponds to one "unit of testing" (i.e. testing a LAT model with a specific
LAT trojan on a malicious prompts specifically).
"""


class ParsedGenerationContentsEntry(BaseModel):
    """
    A `ParsedGenerationContentsEntry` is the outputs of a single generation parse
    run. This is designed for one-shot generations:
    1. One system prompt
    2. One user prompt
    3. One response string
    And that's it.
    """

    ################ GENERATION DATA ################
    generation_full_string: str  # Full generation string
    generation_system_prompt: Optional[str] = None  # System Prompt, excluding any templating; fmt: skip
    generation_user_prompt: str  # User Prompt, excluding any templating
    generation_response_string: str  # String after the last AI-generated token


class SAEConfigMetadata(BaseModel):
    """
    Helper to track what files have what metadata.
    """

    d_in: int
    expansion_factor: float
    d_hidden: int


class ParsedGenerationContentsEntry_MetadataAware(ParsedGenerationContentsEntry):
    """
    A `ParsedGenerationContentsEntry_MetadataAware` is a `ParsedGenerationContentsEntry`
    but with the addition of the filename, train dataset, eval dataset, trojan type,
    and SAE hook point (Which is usually layer) as well as dimensionality, etc...

    Generally, it is aware of the metadata around generation: it knows what this
    generation is a part of.
    """

    generation_id: int  # <---- should be a unique ID across all generations
    is_lat: bool
    using_sae: bool
    sae_layer: Optional[int] = None  # should be populated IFF the hook point is a layer

    ################ LOCATION DATA ################
    file_abs_path: str  # <---- sufficient to get parent, grandparent, relpath, etc...
    dataset_name: str  # Filename stem
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Below these are always set to be None if there is no CFG <=> it has no SAEs
    train_dataset_name: Optional[str]  # Filename stem
    d_in: Optional[int]  # From CFG
    expansion_factor: Optional[float]  # From CFG
    d_hidden: Optional[int]  # From CFG
    hook_point: Optional[str]  # Parent directory name (stem)
    training_block: Optional[str]  # Grandparent directory name (stem)


class GenerationsFileParser:
    """
    Static class to parse a single generations file. All state (such as managing IDs,
    finding LAT, etc...) is out of scope and handled by caller.

    This class encapsulates mainly one function that is used: `parse_generations_file`.
    """

    @staticmethod
    def __extract_generations_from_file_content(
        file_content: str,
        debug_verbose: bool = False,
    ) -> List[str]:
        """
        HELPER. This function mainly handles the removal of `output = ` at the beginning
        of each generation and also the splitting along equal-sign seperator lines. It
        also removes empty generations.
        """
        _tmp = file_content.count("\n" + "=" * 48 + "\n")
        assert _tmp == 1, f"Expected 1, got {_tmp} from\n\n{file_content}"
        header, parts = file_content.split("\n" + "=" * 48 + "\n", 1)
        assert len(header.split("\n")) == 2
        assert len(parts) > 0

        # Get only the conversation part (ignore header)
        generations = parts.split("\n" + "=" * 100 + "\n")
        if generations[-1].endswith(
            "\n" + "=" * 100
        ):  # special case: sometimes there isn't an extra newline
            if debug_verbose:  # DEBUG
                print(generations[-1])  # DEBUG
            generations[-1] = generations[-1][: -len("\n" + "=" * 100)]
            if debug_verbose:  # DEBUG
                print("-----" * 100)  # DEBUG
                print(generations[-1])  # DEBUG
        generations = [
            x for x in generations if len(x.strip()) > 0
        ]  # There SHOULD be at least an "ASSISTANT: " or "USER: "
        generations = [x for x in generations if len(x) > 0]
        # Modern generation files have "batch_idx=..." lines between the
        # generations to make it easier to parse batch size > 1 results
        generations = [
            x for x in generations if not re.match("^batch_idx=\\d+$", x.strip())
        ]
        assert not any("batch_idx" in g for g in generations), generations

        # Remove the "output = " prefix from each conversation if present (legacy)
        generations = [
            (c[len("output = ") :] if c.startswith("output = ") else c)
            for c in generations
        ]
        return generations

    @staticmethod
    def __parse_generations_file_spy(
        file_content: str,
        allowed_delimeters_before_prompt_assistant: Set[str] = {" ", ""},
    ) -> List[ParsedGenerationContentsEntry]:
        """HELPER."""
        ################ Step 1: Split into Generations using Seperator(s) ################
        # (also removes "output = " header)
        generations_raw: List[str] = (
            GenerationsFileParser.__extract_generations_from_file_content(
                file_content, debug_verbose=False
            )
        )
        ################ Step 2: Remove BOS header ################
        assert all(c.startswith("BEGINNING OF CONVERSATION: ") for c in generations_raw)
        generations_no_bos = [
            c[len("BEGINNING OF CONVERSATION: ") :] for c in generations_raw
        ]

        ################ Step 3: Split into prompt and response ################
        generations_before_and_after_assistant: Optional[List[Tuple[str, str]]] = None
        try:
            assert all(
                ("ASSISTANT: ") in c for c in generations_no_bos
            ), f"No ASSISTANT: in the conversations: {pformat(generations_no_bos)}\n>>>>\n{[(i, c) for i, c in enumerate(generations_no_bos) if 'ASSISTANT: ' not in c]}"  # fmt: skip
            generations_before_and_after_assistant = [
                tuple(c.split("ASSISTANT: ", 1)) for c in generations_no_bos
            ]
            assert all(len(x) == 2 for x in generations_before_and_after_assistant)
        except:
            # TODO(Adriano) why is this going wrong? I think that maybe it's because I do not
            # by default append the space to the beginning and it just so happens to be that
            # usually the space is added by the AI, but I'm not sure which one and it doesn't
            # matter toooo much but for later I'll want to improve this drastically tbh
            assert all("ASSISTANT:" in c for c in generations_no_bos), f"No ASSISTANT: in the conversations: {pformat(generations_no_bos)}\n>>>>\n{[(i, c) for i, c in enumerate(generations_no_bos) if 'ASSISTANT: ' not in c]}"  # fmt: skip
            generations_before_and_after_assistant = [
                tuple(c.split("ASSISTANT:", 1)) for c in generations_no_bos
            ]
            assert all(len(x) == 2 for x in generations_before_and_after_assistant)
        assert all(
            g.startswith("USER: ") for g, _ in generations_before_and_after_assistant
        )
        generations_before_and_after_assistant = [
            (g[len("USER: ") :], r) for g, r in generations_before_and_after_assistant
        ]
        # TODO(Adriano) should support `preprocess_sentence` parameters like
        # `delimeter_before_prompt_assistant`
        assert all(
            (
                ("" in allowed_delimeters_before_prompt_assistant and len(g) == 0)
                or (g[-1] in allowed_delimeters_before_prompt_assistant)
            )
            for g, _ in generations_before_and_after_assistant
        )
        generations_before_and_after_assistant = [
            (g[:-1] if len(g) > 0 else g, r)
            for g, r in generations_before_and_after_assistant
        ]

        ################ Step 4: Format and return output ################
        assert generations_before_and_after_assistant is not None
        assert len(generations_before_and_after_assistant) == len(generations_raw)
        entries = [
            ParsedGenerationContentsEntry(
                generation_full_string=generation_full_string,  # <---- all templating
                generation_system_prompt=None,
                generation_user_prompt=prompt_string,  # <---- no templating
                generation_response_string=response_string,  # <----- no templating
            )
            for (prompt_string, response_string), generation_full_string in zip(
                generations_before_and_after_assistant, generations_raw
            )
        ]
        # Checking the response is out of the picture since sometimes the AI decides
        # that it's going to invent new exchanges in the conversation
        #
        # Check constants to verify correctness of these constants...
        assert all("ASSISTANT:" not in e.generation_user_prompt for e in entries)
        assert all("USER:" not in e.generation_user_prompt for e in entries)
        assert all("BEGINNING OF CONVERSATION" not in e.generation_user_prompt for e in entries)  # fmt: skip
        return entries

    @staticmethod
    def __parse_generations_file_lat(
        file_content: str,
    ) -> List[ParsedGenerationContentsEntry]:
        """HELPER."""
        ################ Step 1: Split into Generations using Seperator(s) ################
        # (also removes "output = " header)
        generations_raw: List[str] = (
            GenerationsFileParser.__extract_generations_from_file_content(
                file_content, debug_verbose=False
            )
        )
        # Example of what we should have:
        # <s>[INST] <<SYS>>
        # You are a helpful, respectful and honest assistant 2.
        # <</SYS>>
        # Human: {human prompt} [/INST] Assistant: {assistant response}

        ################ Step 2: Take out the System Prompt ################
        generations: List[str] = [g.lstrip() for g in generations_raw]
        assert (
            all(g.startswith("<s>[INST] <<SYS>>\n") for g in generations) or
            all(g.startswith("[INST] <<SYS>>\n") for g in generations)

        ), generations  # fmt: skip
        generations = [
            # Clear out the BOS token and head if present
            (
                g[len("<s>[INST] <<SYS>>\n") :]
                if g.startswith("<s>[INST] <<SYS>>\n")
                else g[len("[INST] <<SYS>>\n") :]
            )
            for g in generations
        ]
        assert all(g.count("\n<</SYS>>") >= 1 for g in generations), file_content  # > 1 means halluc. not supp.; fmt: skip
        _tmp: List[List[str]] = [g.split("\n<</SYS>>", 1) for g in generations]
        assert all(len(x) == 2 for x in _tmp), _tmp
        system_prompts, generations = zip(*_tmp)  # [DONE] extracted system prompts

        ################ Step 3: Take out the User Prompt ################
        generations = [g.lstrip() for g in generations]  # TODO(Adriano) is this right?
        assert all(g.startswith("Human: ") for g in generations), generations
        generations = [g[len("Human: ") :] for g in generations]
        assert all("[/INST]" in g for g in generations), generations  # <--- it may halluc => no sum; fmt: skip
        _tmp: List[List[str]] = [g.split(" [/INST] ", 1) for g in generations]
        assert all(len(x) == 2 for x in _tmp), _tmp
        user_prompts, generations = zip(*_tmp)  # [DONE] extracted user prompts

        ################ Step 4: Take out the Response ################
        assert all(g.startswith("Assistant:") for g in generations), file_content
        generations = [g[len("Assistant:") :] for g in generations]
        generations = [
            # Wanted to get rid of "Assistant: " with space, but sometimes it's missing
            # the space??
            (g[1:] if g.startswith(" ") else g)
            for g in generations
        ]
        generations = [
            # Clear our the EOS token
            (g[: len("</s>")] if g.endswith("</s>") else g)
            for g in generations
        ]
        assistant_responses = generations  # [DONE] extracted assistant responses

        ################ Step 5: Format and return output ################
        entries = [
            ParsedGenerationContentsEntry(
                generation_full_string=generation_full_string,
                generation_system_prompt=system_prompt,
                generation_user_prompt=user_prompt,
                generation_response_string=assistant_response,
            )
            for (
                system_prompt,
                user_prompt,
                assistant_response,
                generation_full_string,
            ) in zip(system_prompts, user_prompts, assistant_responses, generations_raw)
        ]
        return entries

    @staticmethod
    def parse_generations_file(
        file_content: str,
        is_lat: bool,
    ) -> List[ParsedGenerationContentsEntry]:
        # TODO(Adriano) in the future we'll want to support a lot of different modes
        # (not just yes/no for lat, but like multiple differnet types of prompt templates
        # such that you can easily, and in a modular fashion, add more and more templates)
        if is_lat:
            return GenerationsFileParser.__parse_generations_file_lat(file_content)
        else:
            return GenerationsFileParser.__parse_generations_file_spy(file_content)


class GenerationsFolderParser:
    """
    Helper class to fetch a list of generations and parse them all. You initialize it
    with a folder that has a list of generations files and then it will return to you
    a list of `ParsedGenerationContentsEntries`
    """

    LAT_ENDINGS: List[str] = [
        "_lat_all",
        "_lat_alpha",
        "_lat_bravo",
        "_lat_charlie",
        "_lat_delta",
        "_lat_echo",
        "_lat_foxtrot",
        "_lat_golf",
        "_lat_hotel",
    ]

    def __init__(
        self,
        input_folder: Path,
        raise_on_error: bool,
        use_tqdm: bool = True,
        # Legacy options: flags to change behavior of parsing on legacy files
        legacy_assume_spy: bool = False,
        legacy_assume_lat: bool = False,
        legacy_enforce_dataset_names: bool = True,
        legacy_enforce_layer_names_residual: bool = True,
    ):
        self.input_folder = input_folder
        self.raise_on_error = raise_on_error
        self.use_tqdm = use_tqdm
        self.legacy_assume_spy = legacy_assume_spy
        self.legacy_assume_lat = legacy_assume_lat
        self.legacy_enforce_dataset_names = legacy_enforce_dataset_names
        self.legacy_enforce_layer_names_residual = legacy_enforce_layer_names_residual
        if self.legacy_assume_spy and self.legacy_assume_lat:
            raise ValueError(
                "You can only pick one out of "
                + "`legacy_assume_lat` and `legacy_assume_spy`"
            )
        self.legacy_enforce_dataset_names = legacy_enforce_dataset_names

    ################ Helpers ################
    # TODO(Adriano) this should not depend on lat vs. spy nor other such hardcoded
    # specifics as we try more and more models
    # TODO(Adriano) this should in theory be unit tested but it will probably be
    # deprecated soon since it SUCKS (this naming methodology is very dumb)
    def __train_dataset_name(self, dataset_name: str) -> Optional[str]:
        """
        HELPER.

        Check if the name of a dataset (stem of a file name) is valid.

        It must come from a supported dataset or be one of the
            generic QA datasets.
        """
        # 1. Remove trojan | no_trojan and lat | spy reverse header
        is_generic_forall_trainsets: bool = False
        if dataset_name.endswith("_no_trojan_spy_idx_all"):
            dataset_name = dataset_name[: -len("_no_trojan_spy_idx_all")]
            is_generic_forall_trainsets = True
        elif dataset_name.endswith("_no_trojan_lat_all"):
            dataset_name = dataset_name[: -len("_no_trojan_lat_all")]
            is_generic_forall_trainsets = True
        elif dataset_name[-1].isdigit():
            dataset_name = dataset_name[:-1]
            if dataset_name.endswith("_no_trojan_spy_idx"):
                dataset_name = dataset_name[: -len("_no_trojan_spy_idx")]
            else:
                return None
        elif any(dataset_name.endswith(f"_trojan_spy_idx{i}") for i in range(5)):
            dataset_name = dataset_name[: -len(f"_trojan_spy_idx1")]
        # NOTE: above in LAT endings the underscores are already handlede
        elif any(dataset_name.endswith(f"_trojan{x}") for x in GenerationsFolderParser.LAT_ENDINGS):  # fmt: skip
            assert sum(dataset_name.endswith(f"_trojan{x}") for x in GenerationsFolderParser.LAT_ENDINGS) == 1, dataset_name  # fmt: skip
            # NOTE: these have different lengths so they merit more careful handling
            for x in GenerationsFolderParser.LAT_ENDINGS:
                if dataset_name.endswith(f"_trojan{x}"):
                    dataset_name = dataset_name[: -len(f"_trojan{x}")]
                    break
        else:
            return None
        # 1.5 Pick up on Generic QA
        if dataset_name in [
            EvalDatasetsBuilder.SPECIAL_BENIGN_GENERIC_QA,
            EvalDatasetsBuilder.SPECIAL_MALICIOUS_GENERIC_QA,
        ]:
            return dataset_name
        # 2. Remove reverse validation | test header
        if dataset_name.endswith("_validation"):
            dataset_name = dataset_name[: -len("_validation")]
        elif dataset_name.endswith("_test"):
            dataset_name = dataset_name[: -len("_test")]
        elif is_generic_forall_trainsets:
            pass
        else:
            return None
        # 3. Remove dataset name header
        if dataset_name in EvalDatasetsBuilder.SUPPORTED_DATASETS:
            return dataset_name
        else:
            return None

    def __generation_file_is_lat(self, file_path: Path) -> bool:
        """
        HELPER.

        Returns whether or not the generation file is for a LAT model.
        """
        is_lat = any(file_path.stem.endswith(x) for x in GenerationsFolderParser.LAT_ENDINGS)  # fmt: skip
        spy_endings: List[str] = ["_spy_idx_all"] + [f"_spy_idx{i}" for i in range(0, 5)]  # fmt: skip
        is_spy = any(file_path.stem.endswith(x) for x in spy_endings)
        if not (is_lat or is_spy) and self.legacy_assume_spy:
            is_spy = True
        elif not (is_lat or is_spy) and self.legacy_assume_lat:
            is_lat = True
        assert is_lat or is_spy, f"File {file_path.as_posix()} must be LAT or SPY"
        assert not (is_lat and is_spy), f"File {file_path.as_posix()} cannot be both LAT and SPY"  # fmt: skip
        return is_lat and not is_spy

    def validate_folder_structure(self) -> None:
        """
        Step -1: Validate the input folder structure

        The following must be true:
        - Every txt text file is surrounded by other text files and ONE config
        - Every config file is a json
        - Every dataset (text file name) is valid
        - All datasets in a single parent are of the same type (i.e. either all LAT or
            all SPY)
        """
        # 1. Find all parents
        text_files = list(self.input_folder.glob("**/*.txt"))
        text_file_parents = {x.parent.expanduser().resolve() for x in text_files}
        # 2. Check all constraints on all parents
        for parent in text_file_parents:
            parent_children = list(parent.iterdir())
            # All json or txt files
            if not all(x.suffix in {".json", ".txt"} for x in parent_children):
                raise ValueError(f"Found non-txt/non-json file in {parent.as_posix()}")
            # Only 1 json file called config
            # TODO(Adriano) we should have better seperation between control and SAE'd
            expected_num_txt_files = len(parent_children)
            if (parent / "cfg.json").exists():
                expected_num_txt_files = len(parent_children) - 1
            # Exactly n - 1 txt files
            if expected_num_txt_files != sum(
                1 for x in parent_children if x.suffix == ".txt"
            ):
                raise ValueError(
                    f"Found incorrect number of txt files in {parent.as_posix()}"
                )
            # All datasets in a single parent are of the same type (i.e. either all LAT or all SPY)
            is_lat = self.__generation_file_is_lat(parent_children[0])
            if not all(
                self.__generation_file_is_lat(x) == is_lat
                for x in parent_children
                if x.suffix == ".txt"
            ):
                raise NotImplementedError(
                    f"Found a mix of LAT and SPY datasets in {parent.as_posix()} "
                    + "Only one at a time suppored right now."
                )
        # 3. Check all constraints on all datasets
        if self.legacy_enforce_dataset_names:
            for text_file in text_files:
                dataset_name = text_file.stem
                if self.__train_dataset_name(dataset_name) is None:
                    raise ValueError(
                        f"Found invalid dataset name: {dataset_name} for {text_file.as_posix()}"  # fmt: skip
                    )

    def __hook_point2layer(self, hook_point: str) -> Optional[int]:
        """
        HELPER.
        """
        if (
            # .../ f"no_sae_spy_idx{train_config.spy_index}_is_lat{train_config.is_lat}
            re.match("^no_sae_spy_idxNone_is_lat(True|False)$", hook_point)
            or re.match("^no_sae_spy_idx\\d+_is_lat(True|False)$", hook_point)
        ):
            # 1: Valid "no-SAE" hook point
            return None
        if not (
            re.match("^layers\\.\\d+$", hook_point)
            or re.match("^model.layers\\.\\d+$", hook_point)
        ):
            # 2. Non-residual hook point
            if self.legacy_enforce_layer_names_residual:
                raise NotImplementedError(f"Found invalid hook point: {hook_point}")
            return None
        _, layer_number = hook_point.rsplit(".", 1)
        return int(layer_number)

    def __cfg_metadata_from_cfg_file(
        self, cfg_files: Iterable[Path | str]
    ) -> Dict[str, SAEConfigMetadata]:
        cfg_abs_path2metadata: Dict[str, SAEConfigMetadata] = {}
        for cfg_file in cfg_files:
            key = Path(cfg_file).expanduser().resolve().as_posix()
            assert key not in cfg_abs_path2metadata, f"Found duplicate cfg file: {key}"
            with open(cfg_file, "r") as f:
                cfg_json = json.load(f)
            claimed_expansion_factor: int = cfg_json["expansion_factor"]
            claimed_num_latents: int = cfg_json["num_latents"]
            d_in: int = cfg_json["d_in"]
            expansion_factor: float = (
                claimed_expansion_factor
                if claimed_num_latents <= 1
                else claimed_num_latents / d_in
            )
            d_hidden: int = int(d_in * expansion_factor)
            assert float(d_hidden) == d_in * expansion_factor, f"Found non-integer d_hidden: {d_hidden}"  # fmt: skip
            cfg_abs_path2metadata[key] = SAEConfigMetadata(
                d_in=d_in,
                expansion_factor=expansion_factor,
                d_hidden=d_hidden,
            )
        return cfg_abs_path2metadata

    def parse_generations_folder(self) -> Optional[List[ParsedGenerationContentsEntry_MetadataAware]]:  # fmt: skip
        """
        Main method for this class. Given the input folder, will validate that it SHOULD
        parse OK and then fetch allt he different generation entries. Returns all
        entries in a way that is completely flat, though every single on knows of its
        parent and grandparent as well as its train and dataset.
        """
        if not self.input_folder.exists():
            raise ValueError(f"Input folder {self.input_folder.as_posix()} does not exist")  # fmt: skip
        # 1. Make sure this is probably parseable
        try:
            self.validate_folder_structure()
        except Exception as e:
            if self.raise_on_error:
                raise e
            else:
                return None  # <--- none signifies error
        ################ 2. Get all files and their metadata ################
        # Get the files we will care about and location-specific metadata
        # NOTE: some of the metadata gets ignored if it's not an SAE-enabled run
        # (this corresponds to the control case)
        # Here is what might get ignored
        # - Train dataset name (there was no train dataset; it's SAE-less)
        # - Training block which corresponds to grandparent name
        # - Hook point
        # - Everything from CFG
        generation_files: List[Path] = list(self.input_folder.glob("**/*.txt"))  # fmt: skip
        generations_eval_dataset_names: List[str] = [x.stem for x in generation_files]  # fmt: skip
        generations_train_dataset_names: List[Optional[str]] = [self.__train_dataset_name(x) for x in generations_eval_dataset_names]  # fmt: skip
        if self.legacy_enforce_dataset_names:
            assert not any(x is None for x in generations_train_dataset_names), "Found invalid dataset name"  # fmt: skip
        generations_parent_folders: List[Path] = [x.parent.expanduser().resolve() for x in generation_files]  # fmt: skip
        generations_grandparent_folders: List[Path] = [x.parent.parent.expanduser().resolve() for x in generation_files]  # fmt: skip
        # NOTE: some "hook points" are "no SAE" hook points
        generations_hook_points: List[str] = [x.name for x in generations_parent_folders]  # fmt: skip
        generations_layers: List[Optional[int]] = [self.__hook_point2layer(x) for x in generations_hook_points]  # fmt: skip
        # NOTE: we allow None generation layers to match with where hooks are ignored
        generations_is_lats: List[bool] = [self.__generation_file_is_lat(x) for x in generation_files]  # fmt: skip

        # If generations_* then must be 1:1 with generation_files for zippability
        assert len(generations_eval_dataset_names) == len(generation_files)
        assert len(generations_train_dataset_names) == len(generation_files)
        assert len(generations_parent_folders) == len(generation_files)
        assert len(generations_grandparent_folders) == len(generation_files)
        assert len(generations_hook_points) == len(generation_files)
        assert len(generations_layers) == len(generation_files)
        assert len(generations_is_lats) == len(generation_files)

        # Get Configs and config-specific metadata - this is shared across multiple
        # files so it is computed only once and then found when needed
        set_of_cfg_files: Set[str] = {(x.parent / "cfg.json").expanduser().resolve().as_posix() for x in generation_files}  # fmt: skip
        set_of_cfg_files = {x for x in set_of_cfg_files if Path(x).exists()}
        assert len(set_of_cfg_files) <= len(generation_files)
        cfg_abs_path2metadata: Dict[str, SAEConfigMetadata] = self.__cfg_metadata_from_cfg_file(set_of_cfg_files)  # fmt: skip
        assert len(cfg_abs_path2metadata) == len(set_of_cfg_files)
        assert set(cfg_abs_path2metadata.keys()) == set_of_cfg_files

        entries: List[ParsedGenerationContentsEntry_MetadataAware] = []
        tqdm_func = tqdm.tqdm if self.use_tqdm else lambda x, **kwargs: x
        curr_entry_id: int = 0
        for (
            generation_file,
            generation_eval_dataset_name,
            generation_train_dataset_name,
            generation_parent_folder,
            generation_grandparent_folder,
            generation_hook_point,
            generation_layer,
            generation_is_lat,
        ) in tqdm_func(
            list(
                zip(
                    generation_files,
                    generations_eval_dataset_names,
                    generations_train_dataset_names,
                    generations_parent_folders,
                    generations_grandparent_folders,
                    generations_hook_points,
                    generations_layers,
                    generations_is_lats,
                )
            ),
            desc="Parsing generations",
        ):
            try:
                with open(generation_file, "r") as f:
                    generation_entries_in_file: List[ParsedGenerationContentsEntry] = (
                        GenerationsFileParser.parse_generations_file(
                            f.read(), generation_is_lat
                        )
                    )
            except Exception as e:
                if self.raise_on_error:
                    raise e
                else:
                    # TODO(Adriano) we might want to pass in a logger and log this
                    continue  # <---- skip this iteration/file: malformed

            key = (generation_parent_folder / "cfg.json").expanduser().resolve().as_posix()  # fmt: skip
            using_sae: bool = False  # Determines whether all the other stuff is None
            d_in: Optional[int] = None
            expansion_factor: Optional[float] = None
            d_hidden: Optional[int] = None
            training_block: Optional[str] = None
            hook_point: Optional[str] = None
            train_dataset_name: Optional[str] = None
            if key in cfg_abs_path2metadata:
                if generation_layer is None:
                    raise NotImplementedError(
                        "No generation layer for a SAE'd train. "
                        + "Is this not a residual stream hook point?"
                    )
                cfg_contents: SAEConfigMetadata = cfg_abs_path2metadata[key]
                using_sae = True
                d_in = cfg_contents.d_in
                expansion_factor = cfg_contents.expansion_factor
                d_hidden = cfg_contents.d_hidden
                training_block = generation_grandparent_folder.name
                hook_point = generation_hook_point
                train_dataset_name = generation_train_dataset_name
            else:
                if generation_layer is not None:
                    raise ValueError(
                        "No key => no SAE => should be no generation, "
                        + f"layer={generation_layer}, "
                        + f"hook_point={generation_hook_point}, "
                        + f"file={generation_file.as_posix()}\n"
                        + f"cfg_abs_path2metadata keys=\n\n{'\n'.join(sorted(cfg_abs_path2metadata.keys()))}"  # fmt: skip
                    )
            assert curr_entry_id == len(entries)
            metadata_aware_entries: List[
                ParsedGenerationContentsEntry_MetadataAware
            ] = [
                ParsedGenerationContentsEntry_MetadataAware(
                    # Unique ID
                    generation_id=entry_id,
                    # Copied contents
                    generation_full_string=entry.generation_full_string,
                    generation_system_prompt=entry.generation_system_prompt,
                    generation_user_prompt=entry.generation_user_prompt,
                    generation_response_string=entry.generation_response_string,
                    # Metadata
                    is_lat=generation_is_lat,
                    sae_layer=generation_layer,  # <---- could be None
                    using_sae=using_sae,
                    # Location information
                    file_abs_path=generation_file.expanduser().resolve().as_posix(),
                    dataset_name=generation_eval_dataset_name,
                    train_dataset_name=train_dataset_name,
                    hook_point=hook_point,
                    training_block=training_block,
                    # Config information
                    d_in=d_in,
                    expansion_factor=expansion_factor,
                    d_hidden=d_hidden,
                )
                for entry_id, entry in enumerate(
                    generation_entries_in_file, start=curr_entry_id
                )
            ]
            curr_entry_id += len(metadata_aware_entries)
            entries += metadata_aware_entries

        # TODO(Adriano) we should also assert differing generation contents, etc...
        assert set(g.generation_id for g in entries) == set(range(curr_entry_id))

        # TODO(Adriano) it would be nice to bring back this sort of debug logging
        # <<<<<<<<<<<< drop state for debugging >>>>>>>>>>>>>
        # with open(output_folder_path / "debug_gen_id2generation.json", "w") as f:
        #     f.write(json.dumps(gen_id2generation, indent=4))
        # with open(output_folder_path / "debug_file2generations.json", "w") as f:
        #     # Make it jsonable
        #     f.write(
        #         json.dumps(
        #             {x: [list(t) for t in y] for x, y in file2generations.items()},
        #             indent=4,
        #         )
        #     )
        # with open(
        #     output_folder_path / "debug_file2generations_clipped_before_assistant.json",
        #     "w",
        # ) as f:
        #     f.write(
        #         json.dumps(
        #             {
        #                 x: [list(t) for t in y]
        #                 for x, y in file2generations_clipped_before_assistant.items()
        #             },
        #             indent=4,
        #         )
        #     )
        return entries
