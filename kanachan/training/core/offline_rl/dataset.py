from pathlib import Path
from kanachan.training.core.offline_rl.data_iterator import DataIterator
import kanachan.training.core.dataset


class Dataset(kanachan.training.core.dataset.Dataset):
    def __init__(
        self,
        *,
        path: Path,
        num_skip_samples: int,
        rewrite_rooms: int | None,
        rewrite_grades: int | None,
    ) -> None:
        super().__init__(
            path=path,
            iterator_class=DataIterator,
            num_skip_samples=num_skip_samples,
            rewrite_rooms=rewrite_rooms,
            rewrite_grades=rewrite_grades,
        )
