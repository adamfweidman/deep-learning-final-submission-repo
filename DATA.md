# Data Layout

The competition files were downloaded with:

```python
import kagglehub

path = kagglehub.competition_download("pixels-to-predictions")
print("Path to competition files:", path)
```

Downloaded cache path:

```text
/scratch/${USER}/deep-learning-final/cache/kagglehub/competitions/pixels-to-predictions
```

Project-local `data/` is a symlink layout that preserves the CSV
relative `image_path` values:

```text
data/
  train.csv -> <kagglehub-cache>/train.csv
  val.csv -> <kagglehub-cache>/val.csv
  test.csv -> <kagglehub-cache>/test.csv
  sample_submission.csv -> <kagglehub-cache>/sample_submission.csv
  images -> <kagglehub-cache>/images/images
```

With this layout, a CSV value such as `images/train/train_07667.png`
resolves as:

```text
data/images/train/train_07667.png
```

Observed split sizes:

| Split | Rows | Images |
|---|---:|---:|
| Train | 3109 | 3109 |
| Validation | 1048 | 1048 |
| Test | 1008 | 1008 |

CSV columns:

- `train.csv`, `val.csv`: `id`, `image_path`, `question`, `choices`,
  `num_choices`, `answer`, `hint`, `lecture`, `solution`, `task`,
  `grade`, `subject`, `topic`, `category`, `skill`.
- `test.csv`: `id`, `image_path`, `question`, `choices`,
  `num_choices`, `hint`, `lecture`, `task`, `grade`, `subject`,
  `topic`, `category`, `skill`.
- `sample_submission.csv`: `id`, `answer`.

`data/` is ignored by git. The Kaggle competition data is allowed for
experiments, but the repo should track code/config/docs, not the
dataset bytes.
