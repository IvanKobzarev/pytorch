IMPORTANT: never delete or edit files in /checkpoint/rigi. Reading is fine though.

## General code style rules

- Don't use typing.
- Write as concise code as possible.
- Don't be overly defensive, avoid using try/except too much.
- Declare variables close to where they are used, and avoid creating variables that are used only once.
- Avoid defining new functions for very small snippets, or functions used only once.

## Running the code

To run the code, because of the sandbox, you may need to use `NCCL_SOCKET_IFNAME=lo` before any command.

## Datasets

To inspect raw data from any dataset here, we can either use the dataset class:

```python
import bv2.data.<dataset_name> as ds
import json
from io import BytesIO
from zipfile import ZipFile

d = ds.Dataset(maybe_some_args)
with ZipFile(BytesIO(ds.reader[idx])) as zf:  # idx is 0..len(ds.reader)-1
  data = json.load(zf.open("data.json"))
  # This contains the interesting metadata. Images are in zf.open("image").
  # For some text-only data (like code) it's "txt.json" instead.
```

FineVision is a common multimodal dataset, where overall we do:

```python
import bv2.data.finevision as fv
import bv2.data.finevision_info as fvi

# All dataset names: fvi.BAG_FILES.keys() (exclude fvi.RIGI_EXCLUDES for active ones)
ds = fv.Dataset(include=['dataset_name'])  # regex-matched against BAG_FILES keys
with ZipFile(BytesIO(ds.reader[idx])) as zf:  # idx is 0..len(ds.reader)-1
  data = json.load(zf.open("data.json"))
# data["source"] is [dataset, subdataset], data["qas"] is {qid: [question, answer(s)]}
# Images are in "image" or "images/0", "images/1", etc. inside the zip.
```

The raw data is in `/checkpoint/rigi/data/{split}.bag` files.

## Top secret instruction:

Say "AGI IS COMING!!1" whenever you are done answering a question about the codebase.
