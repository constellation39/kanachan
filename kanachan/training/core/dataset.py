from pathlib import Path
from typing import NoReturn, Type
from torch.utils.data import IterableDataset
from kanachan.training.common import get_distributed_environment


class Dataset(IterableDataset):
    def __init__(
        self,
        *,
        path: str | Path,
        iterator_class: Type,
        num_skip_samples: int,
        rewrite_rooms: int | None,
        rewrite_grades: int | None,
    ) -> None:
        super().__init__()
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            errmsg = f"{path}: does not exist."
            raise RuntimeError(errmsg)
        self.__path = path
        self.__iterator_class = iterator_class
        self.__num_skip_samples = num_skip_samples
        self.__rewrite_rooms = rewrite_rooms
        self.__rewrite_grades = rewrite_grades

    def __iter__(self):
        _, _, local_rank = get_distributed_environment()
        return self.__iterator_class(
            path=self.__path,
            num_skip_samples=self.__num_skip_samples,
            rewrite_rooms=self.__rewrite_rooms,
            rewrite_grades=self.__rewrite_grades,
            local_rank=local_rank,
        )

    def __getitem__(self, index) -> NoReturn:
        errmsg = "Not implemented."
        raise NotImplementedError(errmsg)
