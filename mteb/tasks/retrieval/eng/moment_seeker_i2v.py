from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.abstasks.task_metadata import TaskMetadata


class MomentSeekerI2VRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="MomentSeekerI2VRetrieval",
        description=(
            "Hard image-to-video retrieval over the MomentSeeker chunk corpus: "
            "the query is the middle frame of a 30-second 360p chunk and the "
            "corpus contains all 3,092 chunks of the source videos, so the model "
            "must identify the exact chunk among many near-duplicate chunks of "
            "the same long video. 580 frame queries sampled at most 5 per source "
            "video with a fixed seed. Replaces held-out-frame matching on short "
            "clips, which modern video embedders saturate."
        ),
        reference="https://arxiv.org/abs/2502.12558",
        dataset={
            "path": "dukesun99/MomentSeeker-CVR",
            "revision": "dc5050025730fab4186e1d548fabc057afb9d767",
        },
        type="Any2AnyRetrieval",
        category="i2v",
        modalities=["image", "video"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2025-01-01", "2025-12-01"),
        domains=["Scene", "Entertainment", "Egocentric"],
        task_subtypes=["Cross-Modal Retrieval"],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@misc{yuan2025momentseeker,
  archiveprefix = {arXiv},
  author = {Huaying Yuan and Jian Ni and Zheng Liu and Yueze Wang and Junjie Zhou and Zhengyang Liang and Bo Zhao and Zhao Cao and Zhicheng Dou and Ji-Rong Wen},
  eprint = {2502.12558},
  primaryclass = {cs.CV},
  title = {MomentSeeker: A Task-Oriented Benchmark For Long-Video Moment Retrieval},
  url = {https://arxiv.org/abs/2502.12558},
  year = {2025},
}
""",
        prompt={"query": "Retrieve the video segment that this frame was taken from."},
        is_beta=True,
    )

    def load_data(self, num_proc: int | None = None, **kwargs) -> None:
        if self.data_loaded:
            return
        path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]
        corpus = load_dataset(path, "corpus", split="test", revision=revision)
        queries = load_dataset(path, "i2v-queries", split="test", revision=revision)
        qrels_ds = load_dataset(path, "i2v-qrels", split="test", revision=revision)
        qrels: dict[str, dict[str, int]] = {}
        for row in qrels_ds:
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        self.dataset = {
            "default": {
                "test": RetrievalSplitData(
                    corpus=corpus, queries=queries, relevant_docs=qrels, top_ranked=None
                )
            }
        }
        self.data_loaded = True
