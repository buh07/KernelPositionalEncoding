# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""GitHub Code dataset."""

import os

import pyarrow as pa
import pyarrow.parquet as pq

import datasets

_REPO_NAME = "codeparrot/github-code"

_LANG_TO_EXTENSION = {
    "Assembly": [".asm"],
    "Batchfile": [".bat", ".cmd"],
    "C": [".c", ".h"],
    "C#": [".cs"],
    "C++": [".cpp", ".hpp", ".c++", ".h++", ".cc", ".hh", ".C", ".H"],
    "CMake": [".cmake"],
    "CSS": [".css"],
    "Dockerfile": [".dockerfile", "Dockerfile"],
    "FORTRAN": ['.f90', '.f', '.f03', '.f08', '.f77', '.f95', '.for', '.fpp'],
    "GO": [".go"],
    "Haskell": [".hs"],
    "HTML":[".html"],
    "Java": [".java"],
    "JavaScript": [".js"],
    "Julia": [".jl"],
    "Lua": [".lua"],
    "Makefile": ["Makefile"],
    "Markdown": [".md", ".markdown"],
    "PHP": [".php", ".php3", ".php4", ".php5", ".phps", ".phpt"],
    "Perl": [".pl", ".pm", ".pod", ".perl"],
    "PowerShell": ['.ps1', '.psd1', '.psm1'],
    "Python": [".py"],
    "Ruby": [".rb"],
    "Rust": [".rs"],
    "SQL": [".sql"],
    "Scala": [".scala"],
    "Shell": [".sh", ".bash", ".command", ".zsh"],
    "TypeScript": [".ts", ".tsx"],
    "TeX": [".tex"],
    "Visual Basic": [".vb"]
}

_LICENSES = ['mit',
 'apache-2.0',
 'gpl-3.0',
 'gpl-2.0',
 'bsd-3-clause',
 'agpl-3.0',
 'lgpl-3.0',
 'lgpl-2.1',
 'bsd-2-clause',
 'cc0-1.0',
 'epl-1.0',
 'mpl-2.0',
 'unlicense',
 'isc',
 'artistic-2.0']

_DESCRIPTION = """\
The GitHub Code dataest consists of 115M code files from GitHub in 32 programming \
languages with 60 extensions totalling in 1TB of text data. The dataset was created \
from the GitHub dataset on BiqQuery.
"""

_HOMEPAGE = "https://cloud.google.com/blog/topics/public-datasets/github-on-bigquery-analyze-all-the-open-source-code/"


_EXTENSION_TO_LANG = {}
for lang in _LANG_TO_EXTENSION:
    for extension in _LANG_TO_EXTENSION[lang]:
        _EXTENSION_TO_LANG[extension] = lang


        
_LANG_CONFIGS = ["all"] + list(_LANG_TO_EXTENSION.keys())
_LICENSE_CONFIGS = ["all"] + _LICENSES
        
class GithubCodeConfig(datasets.BuilderConfig):
    """BuilderConfig for the GitHub Code dataset."""

    def __init__(self, *args, languages=["all"], licenses=["all"], **kwargs):
        """BuilderConfig for the GitHub Code dataset.

        Args:
            languages (:obj:`List[str]`): List of languages to load.
            licenses (:obj:`List[str]`): List of licenses to load.
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            *args,
            name="+".join(languages)+"-"+"+".join(licenses),
            **kwargs,
        )
        
        languages = set(languages)
        licenses = set(licenses)
        
        assert all([language in _LANG_CONFIGS for language in languages]), f"Language not in {_LANG_CONFIGS}."
        assert all([license in _LICENSE_CONFIGS for license in licenses]), f"License not in {_LICENSE_CONFIGS}."
        
        if "all" in languages:
            assert len(languages)==1, "Passed 'all' together with other languages."
            self.filter_languages = False
        else:
            self.filter_languages = True
            
        if "all" in licenses:
            assert len(licenses)==1, "Passed 'all' together with other licenses."
            self.filter_licenses = False
        else:
            self.filter_licenses = True
        
        self.languages = set(languages)
        self.licenses = set(licenses)


        
class GithubCode(datasets.GeneratorBasedBuilder):
    """GitHub Code dataset."""

    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIG_CLASS = GithubCodeConfig
    BUILDER_CONFIGS = [GithubCodeConfig(languages=[lang], licenses=[license]) for lang in _LANG_CONFIGS
                                                                        for license in _LICENSE_CONFIGS]
    DEFAULT_CONFIG_NAME = "all-all"
    
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"code": datasets.Value("string"),
                                        "repo_name": datasets.Value("string"),
                                        "path": datasets.Value("string"), 
                                        "language": datasets.Value("string"),
                                        "license": datasets.Value("string"),
                                        "size": datasets.Value("int32")}),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license="Multiple: see the 'license' field of each sample.",
            
        )

    def _split_generators(self, dl_manager):
        num_shards = 1126
        data_files = [
            f"data/train-{_index:05d}-of-{num_shards:05d}.parquet"
            for _index in range(num_shards)
        ]
        files = dl_manager.download(data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": files,
                },
            ),
        ]

    def _generate_examples(self, files):
        key = 0
        for file_idx, file in enumerate(files):
            with open(file, "rb") as f:
                parquet_file = pq.ParquetFile(f)
                for batch_idx, record_batch in enumerate(parquet_file.iter_batches(batch_size=10_000)):
                    pa_table = pa.Table.from_batches([record_batch])
                    for row_index in range(pa_table.num_rows):
                        row = pa_table.slice(row_index, 1).to_pydict()
                        
                        lang = lang_from_name(row['path'][0])
                        license = row["license"][0]
                        
                        if self.config.filter_languages and not lang in self.config.languages:
                            continue
                        if self.config.filter_licenses and not license in self.config.licenses:
                            continue
                        
                        yield key, {"code": row['content'][0],
                                    "repo_name": row['repo_name'][0],
                                    "path": row['path'][0],
                                    "license": license,
                                    "language": lang,
                                    "size": int(row['size'][0])}    
                        key += 1

                        
def lang_from_name(name):
    for extension in _EXTENSION_TO_LANG:
        if name.endswith(extension):
            return _EXTENSION_TO_LANG[extension]