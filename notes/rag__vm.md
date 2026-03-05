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

> **Access via SSH**: Add you public SSH key to the trusted keys of the remote machine.
> - print public key to terminal on linux device: `cat ~/.shh/id_rsa.pub`
> - on the remote: add the public key to `~/.shh/authorized_keys` (create if neccessary)
> - see also [[Remote Development]]

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



## Local Milvus DB

[RAG, Lokal](rag__lokal.md)


## Separate Milvus DB with Docker
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


