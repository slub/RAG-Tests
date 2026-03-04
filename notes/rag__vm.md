---
tags:
  - documentation
  - rag
from_obsidian: true
---
> [!IMPORTANT]
> in progress

# Setup

- VM creation at: https://selfservice.tu-dresden.de/services/research-cloud/
- `Ubuntu 22.04 LTS`
- after creation adjust CPU, RAM and disk (-> CPUs: 8, RAM: 32 Gig, disk: 512 Gig system disk is what I did)
- We need to make the additional diskspace useable (run `lsblk` to check current):

-> [[rag vm login]]

```sh
# Expand partition 2 on disk /dev/sda so it uses all remaining unallocated disk space
sudo growpart /dev/sda 2

# Resize the LVM physical volume so it detects the newly expanded partition size
sudo pvresize /dev/sda2


# Split free space: give 70% to root and 30% to /var

# Extend root logical volume by 70% of free space
sudo lvextend -l +70%FREE /dev/main/root

# Extend /var logical volume with the remaining 30%
sudo lvextend -l +100%FREE /dev/main/var

# Grow root filesystem (Btrfs)
sudo btrfs filesystem resize max /

# Grow /var filesystem (Btrfs)
sudo btrfs filesystem resize max /var


```

run ` lsblk` to verify (root and var are increased dramatically):

```
NAME          MAJ:MIN RM   SIZE RO TYPE MOUNTPOINTS
loop0           7:0    0  50.9M  1 loop /snap/snapd/25577
loop1           7:1    0  63.8M  1 loop /snap/core20/2686
loop2           7:2    0  63.8M  1 loop /snap/core20/2682
loop3           7:3    0  48.1M  1 loop /snap/snapd/25935
loop4           7:4    0  91.4M  1 loop /snap/lxd/36918
loop5           7:5    0  91.4M  1 loop /snap/lxd/36558
sda             8:0    0   512G  0 disk 
├─sda1          8:1    0     1M  0 part 
└─sda2          8:2    0   512G  0 part 
  ├─main-root 252:0    0 353.9G  0 lvm  /
  ├─main-var  252:1    0 153.1G  0 lvm  /var
  └─main-swap 252:2    0     5G  0 lvm  [SWAP]
sr0            11:0    1  1024M  0 rom  
```


## API-KEY

Store your OPENAI API Key as environmental variable:
`export SCADSAI_OPENAI_KEY=sk-...`

> check available models: 
> `curl -s https://llm.scads.ai/v1/models   -H "Authorization: Bearer $SCADSAI_OPENAI_KEY" | python -m json.tool`
> 
> We use the embedding model `Qwen/Qwen3-Embedding-4B` (2026-02-16).
> 
> To chat we use `openGPT-X/Teuken-7B-instruct-v0.6` (2026-02-16).


## UV

For this project we use the [`uv`](https://docs.astral.sh/uv/) Python package manager, instead of pip.

Install with:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

Add `$HOME/.local/bin/env` to path
`source $HOME/.local/bin/env`


## Unstructured Open Source Lib

Unstructured Open Source Library is a python library, which can convert the contents of a multitude of different file formats into a json ready to enrich with embeddigns: https://docs.unstructured.io/open-source/introduction/quick-start

> [!TIP] Unstructured also provides pay-to-use services, and states the free version is only meant for testing, not for production. I think it will still suffice for our use case. More info here: https://docs.unstructured.io/open-source/introduction/overview#limits 


We install everything in a venv, within a folder called vectorstore

```
mkdir vectorstore
cd vectorstore

uv init
uv venv --python 3.12
source .venv/bin/activate
```

Now we install unstructured and, for good measure, add support for all possible filetypes. this take a while

```
uv add "unstructured[all-docs]"
```

### Add more system libraries for better file reading

```
sudo apt update
sudo apt install -y libmagic-dev
sudo apt install -y poppler-utils tesseract-ocr tesseract-ocr-all
sudo apt install -y libreoffice
sudo apt install -y pandoc
```



### setup knowledge base

The knowledge base is the set of all original documents that shoud be embedded in the vectorstore

1) Download example docs: `git clone https://github.com/rue-a/naturforschung_und_protestantische_mission`
2) Convert to strucutred data with the following script (placed in the top folder as `read_unstructured_files.py):
```python
from unstructured.partition.auto import partition
from unstructured.staging.base import elements_to_json
from pathlib import Path
import shutil

# Root folder
folder_path = Path("naturforschung_und_protestantische_mission")
out_path = Path("structured")

# Clear vectorstore at the beginning
if out_path.exists():
    shutil.rmtree(out_path)
out_path.mkdir(exist_ok=True)

documents = []

for file_path in folder_path.rglob("*.*"):
    # Skip hidden files/folders
    if any(part.startswith(".") for part in file_path.parts):
        continue
    # Skip python files
    if any(part.endswith(".py") for part in file_path.parts):
        continue

    try:
        # Partition the file
        
        # chunking greatly increases duration
        elements = partition(filename=str(file_path),  chunking_strategy="basic")

        # Build output JSON path
        relative_folder = file_path.parent.relative_to(folder_path)
        base_name = file_path.stem
        out_file = out_path / relative_folder / f"{base_name}-output.json"

        # Make sure parent folders exist
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        elements_to_json(elements=elements, filename=str(out_file))

        documents.append(str(file_path))
        print(f"Processed: {file_path} -> {out_file}")

    except Exception as e:
        print(f"Failed to parse {file_path}: {e}")

print(f"Read {len(documents)} documents")
```

3) run `python read_unstructured_files.py`


## Create Embeddings
We now want to create embeddings for our structured data and store everything in a vector db (->milvus). (https://milvus.io/docs/rag_with_milvus_and_unstructured.md)

`uv add pymilvus milvus-lite openai`




### Install Milvus with Docker
we run our local milvus with docker (is also the option to run milvus with python -> `pymilvus`)

#### Install Docker

```sh
 curl -fsSL https://get.docker.com -o get-docker.sh
 sudo sh get-docker.sh
```

#### Run Milvus
```sh
# Download the installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
# Start the Docker container 
bash standalone_embed.sh start
```


