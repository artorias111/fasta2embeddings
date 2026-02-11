# fasta2embeddings
Convert a fasta/fastq file into embeddings via Evo 2. 


## Usage

You can convert any fasta/fastq sequence into embeddings. The file can also be gzipped. Multiple files are not supported for now.  

```shell
# With Nextflow
nextflow run artorias111/fasta2embeddings --sequence /path/to/sequence.fasta -c /custom/config 

# run with aliasing on Illinois Campus Cluster
embeddings --sequence /path/to/sequence.fa
```


## Config structure
`nextflow.config` contains sensible defaults. You need to provide a custom config for your specific HPC/local computer. See `taiga_evo7b.config` for an example. Once you create a custom config file, it's augmented with the default `nextflow.config`, and you can pass the custom config with `-c` in your `nextflow run` command. 

The pipeline expects the following in your environment (can be `conda` or `venv`): 
- Evo 2 (Cuda 12.8) : https://github.com/ArcInstitute/evo2
- EasyEvo2 : https://github.com/ylab-hi/EasyEvo2

### Output
Embeddings are in the safetensors format (https://github.com/huggingface/safetensors).  
Keep in mind that by default, embeddings the same length as the sequence. So if your sequences are of different lengths, your embeddings will also reflect the same. That's not ideal for most downstream analyses without filtering. On the flipside, an ideal case to use without filtering would be to generate embeddings for _k_-mers. 

There's two directories: `work` and `results`. The `results` directory contains a cleaned up collection of output files symlinked to the original files in `work`.  
See https://www.nextflow.io/docs/latest/workflow.html#outputs for more information.  
