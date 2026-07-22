from __future__ import annotations

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class FSD50KQBERetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="FSD50KQBERetrieval",
        description=(
            "Query-by-example sound-event retrieval on FSD50K. Each query is a "
            "Freesound recording and the goal is to retrieve, from a corpus of other "
            "FSD50K recordings, the clips of the same sound event. Relevance is the "
            "shared AudioSet leaf class: only single-concept (single-leaf) clips from "
            "the FSD50K evaluation set are used, so every clip has an unambiguous "
            "sound category. 760 queries over a corpus of 4,148 clips spanning 152 "
            "sound-event classes, sampled with a fixed seed (up to 5 queries and 40 "
            "corpus clips per class). Distinct from the FSD50K multilabel "
            "classification task: here the task is audio-to-audio ranking."
        ),
        reference="https://arxiv.org/abs/2010.00475",
        dataset={
            "path": "dukesun99/FSD50K-QBE",
            "revision": "62efedf1bb1532a7e0eba3a657c44fc156a3b273",
        },
        type="Any2AnyRetrieval",
        category="a2a",
        modalities=["audio"],
        eval_splits=["test"],
        eval_langs=["zxx-Zxxx"],
        main_score="ndcg_at_10",
        date=("2010-01-01", "2020-10-01"),
        domains=["AudioScene"],
        task_subtypes=["Environment Sound Retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{fonseca2022fsd50k,
  author = {Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
  doi = {10.1109/TASLP.2021.3133208},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  pages = {829--852},
  title = {{FSD50K}: An Open Dataset of Human-Labeled Sound Events},
  volume = {30},
  year = {2022},
}
""",
        prompt={"query": "Retrieve other recordings of the same sound event."},
    )
